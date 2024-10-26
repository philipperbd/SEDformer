ssh slurm-ext
script à lancer: SEDformer/slurm_jobs/train.sh
surveiller avec: tail -f

erreur actuelle: ALLOCATION MEMOIRE

piste:

---

# TODO.md

## Objectif
Optimiser la consommation mémoire de `SEDformer` et résoudre les erreurs d'allocation mémoire rencontrées lors de l'exécution.

## Étapes à Suivre

### 1. Vérifier la Taille du Modèle
- [ ] Réduire la profondeur de la signature (`depth`) dans les configurations :
  ```python
  configs.signature_depth = 2  # Réduire à 2 pour tester
  ```
- [ ] Réduire la longueur des séquences (`seq_len`) pour limiter la taille des tenseurs :
  ```python
  configs.seq_len = 64  # Par exemple, diminuer à 64 pour tester
  ```
- [ ] Diminuer la taille du batch :
  ```python
  configs.batch_size = 16  # Réduire pour tester
  ```
  
### 2. Ajuster la Taille des Tenseurs
- [ ] Limiter le nombre de caractéristiques dans la couche linéaire pour éviter des dépassements :
  - Modifier `SignatureCorrelation.py` :
    ```python
    max_features = 1024  # Limiter à une taille raisonnable
    in_features = min((in_channels * (in_channels**self.depth - 1)) // (in_channels - 1), max_features)
    self.linear = nn.Linear(in_features, out_channels)
    ```
- [ ] Ajouter des `print` statements pour surveiller les tailles des tenseurs après chaque couche :
  ```python
  print(f"Taille des poids de la couche linéaire : {self.linear.weight.size()}")
  ```

### 3. Surveiller la Mémoire pendant l'Exécution
- [ ] Surveiller la mémoire GPU en temps réel :
  ```bash
  watch -n 2 nvidia-smi
  ```
- [ ] Surveiller la mémoire RAM en temps réel :
  ```bash
  watch -n 2 free -h
  ```

### 4. Ajuster la Complexité du Modèle
- [ ] Réduire le nombre de dimensions dans les couches d'attention :
  ```python
  configs.d_model = 8  # Réduire la dimension pour tester
  ```
- [ ] Diminuer le nombre de couches de l'encodeur et du décodeur :
  ```python
  configs.e_layers = 1  # Une seule couche de l'encodeur
  configs.d_layers = 1  # Une seule couche du décodeur
  ```

### 5. Vérifier l'Utilisation de la Mémoire et les Tailles du Modèle
- [ ] Ajouter une vérification de la mémoire allouée avant et après la création du modèle :
  ```python
  import torch

  # Avant la création du modèle
  print(f"CUDA available: {torch.cuda.is_available()}")
  print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory} bytes")

  # Créez le modèle
  self.model = self._build_model().to(self.device)

  # Après la création du modèle
  print(f"Memory allocated: {torch.cuda.memory_allocated()}")
  print(f"Memory reserved: {torch.cuda.memory_reserved()}")
  ```

### 6. Tester les Modifications et Analyser les Résultats
- [ ] Soumettre à nouveau le job SLURM avec les ajustements :
  ```bash
  sbatch train.sh
  ```
- [ ] Analyser les fichiers de log (`.stdout` et `.stderr`) pour vérifier si la mémoire allouée reste raisonnable.
- [ ] Ajuster les paramètres en fonction des résultats pour trouver la configuration optimale.

## Résumé
- Adaptez progressivement les paramètres pour identifier la source des dépassements de mémoire.
- Surveillez les ressources en temps réel pour ajuster la configuration.
- Limitez la taille des poids et des tenseurs pour rester dans les capacités du système.