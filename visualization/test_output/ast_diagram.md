# Diagrama AST - Análisis de Algoritmo

## Árbol de Sintaxis Abstracta

```mermaid
graph TD
    n1["for_loop: PARA i DESDE 0 HASTA n-1 HACER"] :::loop
    n2["other: arreglo[i] <- i + 1"] 
    n3["statement: FIN PARA"] 
    n4["while_loop: MIENTRAS izquierda <= derech"] :::loop
    n5["assignment: medio <- (izquierda + derech"] 
    n6["if_statement: SI arreglo[medio] == x ENT"] :::condition
    n7["assignment: encontrado <- VERDADERO"] 
    n8["statement: FIN SI"] 
    n9["statement: FIN MIENTRAS"] 


    classDef loop fill:#FFB6C1,stroke:#333,stroke-width:2px,color:#000
    classDef condition fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    classDef call fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    classDef function fill:#FFD700,stroke:#333,stroke-width:2px,color:#000
    classDef return fill:#FFA500,stroke:#333,stroke-width:2px,color:#000

```

## Leyenda de Colores

- Rosa: Bucles (for, while)
- Azul: Condicionales (if)
- Verde: Llamadas a funciones
- Oro: Definiciones
- Naranja: Retorno
