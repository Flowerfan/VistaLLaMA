cd /opt/tiger


sudo apt-get update
sudo apt-get install git-lfs
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xvf Python-3.10.0.tgz
cd Python-3.10.0
sudo ./configure --enable-optimizations
sudo make altinstall

cd /opt/tiger/VideoLLM/
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install httpx==0.23.0
python3.10 -m pip install ninja
python3.10 -m pip install pydantic==1.10.9
python3.10 -m pip install langchain timm omegaconf


cd /mnt/bn/vlp-v6/flash-attention
python3.10 setup.py install --user

python3.10 -m pip install openai==0.28.0

### NextQA dataset
cd /opt/tiger/
unzip /mnt/bn/vlp-lq/NExTVideo.zip .
cd VideoLLM
cp -r /mnt/bn/vlp-v6/VideoLLM/LLaVA-7B-Lightening-v1-1/ .


