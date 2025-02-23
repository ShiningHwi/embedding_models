# 한국어 설치형 임베딩 모델에 대한 벤치마크 데이터셋 테스트 결과

![스크린샷 2025-02-24 081831](https://github.com/user-attachments/assets/de5669dc-d1b6-4e3c-8309-7f1a674361f5)
![스크린샷 2025-02-24 081853](https://github.com/user-attachments/assets/9856b998-c3c4-4f2a-a238-3d01e74b9252)
![스크린샷 2025-02-24 081903](https://github.com/user-attachments/assets/e26cad00-cc3f-4348-8f8c-8c6a38dc5970)

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

1. https://github.com/nlpai-lab 를 참고했습니다.
2. https://arca.live/b/alpaca/119279222 를 참고했습니다.
3. https://github.com/su-park/mteb_ko_leaderboard 를 참고했습니다.
