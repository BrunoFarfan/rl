# Pasos para reproducir los experimentos del informe:

## MDPs/Main.py

### Pregunta d)

Lo único que se necesita es el siguiente bloque de código en `main.py`:

```python

if __name__ == "__main__":
    play_problem('grid', uniform=True)
    play_problem('cookie', uniform=False)
    play_problem('gambler', uniform=False)
```

### Pregunta h)

Para esta pregunta se necesita el siguiente bloque de código en `main.py`:

```python
if __name__ == "__main__":
    play_value_iteration()
```

### Pregunta i)

Para esta pregunta se necesita el siguiente bloque de código en `main.py`:

```python
if __name__ == "__main__":
    analyze_gambler_multiple_optimal_policies(p=0.55)
```

Donde el valor de `p`se puede ir variando según lo que pide el enunciado.


## MonteCarlo/Main.py

### Pregunta j)

Para correr los respectivos experimentos de esta pregunta se necesitan los siguientes bloques de código en `Main.py`:

```python
if __name__ == "__main__":
    returns_cliff = run_experiment_cliff(
        agent_class=MonteCarloControlEveryVisit,
        env_class=CliffEnv,
        epsilon=0.1,
        gamma=1.0,
    )

    print("Retornos Cliff:", returns_cliff)

    returns_bj = run_experiment_blackjack(
        agent_class=MonteCarloControlEveryVisit,
        env_class=BlackjackEnv,
        epsilon=0.01,
        gamma=1.0
    )

    print("Retornos Blackjack:", returns_bj)
```