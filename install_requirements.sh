#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <nom_de_l_environnement>"
    exit 1
fi

env_name=$1

python3.10 -m venv $env_name

source $env_name/bin/activate

pip install -r requirements.txt

echo ""
echo -e "Dépendances installées avec succès dans l'environnement virtuel $env_name à partir de requirements.txt."
echo -e "Pour activer:\n\n"
echo -e "    source $env_name/bin/activate"
echo -e "\n"