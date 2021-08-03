#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J rmpx-test

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=12

## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=10GB

## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00

## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgkhozzy2021a

## Specyfikacja partycji
#SBATCH -p plgrid-testing

## Plik ze standardowym wyjściem
#SBATCH --output="output.out"

## Plik ze standardowym wyjściem błędów
#SBATCH --error="error.err"

module load plgrid/tools/python-intel/3.5.3

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR/slurm

./execute.sh
