# Instrucciones para ejecutar los experimentos

El siguiente ejecuta todos los experimentos:

```
python Main.py
```

## **Opciones para deshabilitar partes específicas**
Las siguientes flags para desactivar partes del experimento:

- `--no-parte-a` deshabilita la ejecución de la parte a) de la tarea
- `--no-parte-c` deshabilita la ejecución de la parte c) de la tarea
- `--no-parte-f` deshabilita la ejecución de la parte f) de la tarea

### **Ejemplo**
Ejecutar solo la parte C:
```
python Main.py --no-parte-a --no-parte-f
```
Ejecutar solo la parte F:
```
python Main.py --no-parte-a --no-parte-c
```

Por defecto, todas las partes están habilitadas.