docker build . -t fclnn
docker run -it -v .:/fclnn --rm -p 8888:8888 fclnn