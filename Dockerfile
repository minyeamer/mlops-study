# 파이썬 3.10 공식 이미지 사용
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 전체 프로젝트 파일 복사
COPY . .

# 파이썬 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 실행 스크립트 권한 부여
RUN chmod +x run_service.sh scripts/*.sh

# 컨테이너 실행 시 run_service.sh 자동 실행
CMD ["/app/run_service.sh"]