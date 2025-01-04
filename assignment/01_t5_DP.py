import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler

import warnings
warnings.filterwarnings("ignore")  # 경고 메시지를 무시하도록 설정

# Hugging Face 모델 다운로드를 위한 HF_HOME 설정
HF_HOME = os.getenv("HF_HOME", "./hf_models")  # 환경 변수로 모델 캐시 경로 지정, 기본값은 ./hf_models
os.environ["HF_HOME"] = HF_HOME  # HF_HOME 환경 변수 등록
os.makedirs(HF_HOME, exist_ok=True)  # 모델 저장 경로가 없으면 생성

# DummyDataset: 랜덤 데이터를 생성하는 더미 데이터셋 클래스
class DummyDataset(Dataset):
    def __init__(self, num_samples=640, num_tokens=256, max_len=256, seed=42):
        super().__init__()
        torch.manual_seed(seed)  # 시드 고정으로 데이터 일관성 유지
        self.num_samples = num_samples
        self.input_ids = torch.randint(0, num_tokens, size=(num_samples, max_len))
        self.decoder_input_ids = torch.randint(0, num_tokens, size=(num_samples, max_len))
        self.labels = torch.randint(0, num_tokens, size=(num_samples, max_len))

    def __len__(self):
        return self.num_samples  # 데이터셋 크기 반환

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터를 반환
        return {
            "input_ids": self.input_ids[idx],
            "decoder_input_ids": self.decoder_input_ids[idx],
            "labels": self.labels[idx],
        }

# 메모리 확인 및 최대 배치 크기 탐색
def find_max_batch_size(device, start_batch_size=2, max_batch_size=1024):
    """
    GPU 메모리 한계 내에서 최대 배치 크기를 탐색하는 함수
    Args:
        device: GPU 장치 객체
        start_batch_size: 탐색 시작 배치 크기
        max_batch_size: 탐색할 최대 배치 크기
    """
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-large", cache_dir=HF_HOME  # Hugging Face 모델 로드
    )
    model.to(device)  # 모델을 GPU로 이동
    model = DataParallel(model)  # DataParallel로 병렬 처리 활성화
    dataset = DummyDataset(num_samples=100)  # 테스트 데이터셋 생성
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD 옵티마이저 설정

    batch_size = start_batch_size
    success_batch_size = batch_size  # 성공한 마지막 배치 크기를 저장

    pbar = tqdm(total=max_batch_size, desc="Finding Max Batch Size", unit="batch")  # 진행 표시줄

    while batch_size <= max_batch_size:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # 데이터 로더 생성

        try:
            torch.cuda.empty_cache()  # GPU 메모리 정리
            for data in dataloader:
                # 데이터를 GPU로 이동
                data = {k: data[k].to(device) for k in data}
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):  # Mixed Precision 사용
                    output = model(**data)
                    loss = torch.mean(output.loss)  # 손실 계산
                loss.backward()  # 그래디언트 계산
                optimizer.step()  # 파라미터 업데이트
                break  # 첫 번째 배치만 처리

            # 진행 상황 업데이트
            pbar.update(batch_size - pbar.n)
            pbar.set_postfix(success_batch_size=batch_size)
            success_batch_size = batch_size  # 현재 배치 크기 저장
            batch_size *= 2  # 배치 크기를 2배로 증가

        except RuntimeError as e:
            # CUDA 메모리 부족 시 처리
            if "CUDA out of memory" in str(e):
                print(f"Batch size {batch_size} failed due to OOM.")
                batch_size //= 2
                break
            else:
                # 예기치 않은 오류 발생 시
                print(f"Unexpected error occurred: {str(e)}")
                raise e

    pbar.close()
    print(f"Maximum batch size: {success_batch_size}")  # 최대 배치 크기 출력
    return success_batch_size

# 학습 함수
def train(device, batch_size, epochs=1):
    """
    모델 학습 함수
    Args:
        device: GPU 장치 객체
        batch_size: 학습에 사용할 배치 크기
        epochs: 학습 에포크 수
    """
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-large", cache_dir=HF_HOME  # Hugging Face 모델 로드
    )
    model.to(device)  # 모델을 GPU로 이동
    model = DataParallel(model)  # DataParallel로 병렬 처리 활성화

    dataset = DummyDataset()  # 데이터셋 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # 데이터 로더
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD 옵티마이저 설정
    scaler = GradScaler()  # Mixed Precision 학습을 위한 GradScaler

    start_time = torch.cuda.Event(enable_timing=True)  # 타이밍 시작 이벤트
    end_time = torch.cuda.Event(enable_timing=True)  # 타이밍 종료 이벤트

    for epoch in tqdm(range(epochs), desc="Epoch Progress", leave=True):  # 에포크 진행 표시줄
        batch_bar = tqdm(dataloader, desc=f"Batch Progress (Epoch {epoch + 1}/{epochs})", leave=False)
        for i, data in enumerate(batch_bar):
            # 데이터를 GPU로 이동
            data = {k: data[k].to(device) for k in data}

            optimizer.zero_grad()
            start_time.record()  # 타이밍 시작
            with torch.amp.autocast(device_type='cuda'):  # Mixed Precision 사용
                output = model(**data)
                loss = torch.mean(output.loss)  # 손실 계산
            scaler.scale(loss).backward()  # Mixed Precision에서 그래디언트 계산
            scaler.step(optimizer)  # 옵티마이저 단계
            scaler.update()  # GradScaler 업데이트
            end_time.record()  # 타이밍 종료
            torch.cuda.synchronize()  # GPU 동기화

            # 처리된 토큰 계산
            batch_tokens = batch_size * data["input_ids"].size(1)
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 밀리초 → 초 변환
            tokens_per_second = batch_tokens / elapsed_time if elapsed_time > 0 else 0  # 초당 처리된 토큰 수 계산

            # 진행 표시줄 업데이트
            batch_bar.set_postfix(
                loss=loss.item(),
                tokens_per_second=f"{tokens_per_second:.2f}",
            )

# GPU 설정 및 최대 배치 크기 탐색 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 설정
    print("Finding maximum batch size...")
    max_batch_size = find_max_batch_size(device=device, start_batch_size=2, max_batch_size=1024)  # 최대 배치 크기 탐색

    print(f"Starting training with batch size {max_batch_size}...")
    train(device=device, batch_size=max_batch_size, epochs=2)  # 학습 실행
