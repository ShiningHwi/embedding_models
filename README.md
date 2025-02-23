# 한국어 설치형 임베딩 모델에 대한 벤치마크 데이터셋 테스트 결과

고려대학교 KURE-v1 임베딩 모델 성능이 가장 좋았습니다 (2025.02.24).

# 실행 방법

1. 벤치마크 실행
python main.py

2. 평가 실행
streamlit run leaderboard.py

# 구조 및 환경 설명

1. 사용한 평가모델 및 벤치마크 데이터셋은 config.py에 있습니다. 
2. 평가모델을 추가하고자 하신다면, MODELS = [] 목록에 입력하시면 됩니다.
2. 벤치마크 데이터셋을 추가하고자 하신다면, data_loaders/__init__.py 에서 부가적인 작업이 요구됩니다.
4. 로컬 환경이어서 배치사이즈를 1로 사용했습니다. 
5. python 3.10, cuda 12.1 

# 참고 자료

1. https://github.com/nlpai-lab를 참고했습니다.
2. https://arca.live/b/alpaca/119279222를 참고했습니다.
3. https://github.com/su-park/mteb_ko_leaderboard를 참고했습니다.
