
docker rm flask

docker run --name flask -p 5000:5000 -v $PWD:/src -e PYTHONPATH=$PYTHONPATH:/src -it flask python3 /src/main.py
