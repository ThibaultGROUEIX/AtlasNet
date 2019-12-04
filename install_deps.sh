#  Original solution via StackOverflow:

# https://gist.github.com/luiscape/19d2d73a8c7b59411a2fb73a697f5ed4
#  Install via `conda` directly.

while read requirement; do conda install --yes $requirement; done < requirements.txt 2>error.log

