conda create -n regnerf python=3.7
conda activate regnerf
pip install -r requirements.txt
pip install --upgrade jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install lpips
pip install ipdb

To fix jax.random error: https://github.com/kingoflolz/mesh-transformer-jax/issues/221
                   pip3 install chex==0.1.2
To fixprotobuf error: pip install protobuf==3.20.1
