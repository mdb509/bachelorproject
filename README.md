
# Using #SAT for Mastermind

### Commands for bachelorproject: Solving Mastermind with #SAT

#### with docker:


```
mkdir -p export
docker build -t bench .
docker run --rm -v "$PWD/export:/export" bench
```

#### without docker:

i used python3.12.3 for this project

clone repos

```
git clone https://github.com/mdb509/bachelorproject.git && \
git clone https://github.com/arminbiere/dualiza.git && \
git clone https://github.com/timpehoerig/master_project.git
```
get latest release at https://github.com/meelgroup/ganak/releases for ganak and unzip ganak-linux-amd64.zip

build bc_enum (blocked_clause_enumeration)

```
cd ../master_project/
git clone https://github.com/arminbiere/cadical.git cadical
```

move to bachelorproject and create venv and install requirements.txt (numpy, matplot)

```
cd ../bachelorproject
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

start main.py that outputs benchmark_all.json for all tests in main.py

```
python3 main.py 
```

generate plots from benchmark_all.json

```
python3 plot/plot.py
```

outputs plots in bachelorproject/results/
done