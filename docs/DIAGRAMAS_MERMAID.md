# 📐 Diagramas en Mermaid — Sistema DermAI Full-Stack Platform

Este documento reúne los 5 diagramas requeridos formateados formalmente en Mermaid.js para su renderizado directo en GitHub o editores Markdown.

---

## 1. Diagrama de Arquitectura de Software

```mermaid
graph TD
    Client[Cliente / Navegador Web] -->|HTTP / JSON| Frontend[Frontend Next.js 14]
    Frontend -->|Modo Claro/Oscuro| UI[Tailwind CSS UI]
    Frontend -->|i18n ES/EN| I18N[i18n Translation Engine]
    Frontend -->|Web Speech API| Voice[Voz & Text Chatbot]
    
    Frontend -->|REST API Calls| FastAPI[Backend FastAPI Python]
    
    FastAPI -->|Autenticación| Auth[Auth Module - JWT]
    FastAPI -->|EDA| EDAModule[Descriptivos & Dist. Clases]
    FastAPI -->|Entrenamiento| TrainEngine[Motor ML: 3 Clásicos + 2 Híbridos]
    FastAPI -->|Cross Validation| CVEngine[5-Fold Stratified CV]
    FastAPI -->|Tuning| TuningEngine[Hyperparameter Optimizer]
    FastAPI -->|Pruebas Robustas| StatsEngine[Engine Estadístico: KS, MW, MP, McNemar, LM]
    FastAPI -->|Exportación| ReportEngine[PDF, Word, Excel Builders]
    
    TrainEngine -->|Guardar .keras / .h5| ModelsDisk[Disco: saved_models/]
    ReportEngine -->|Generar Archivos| ReportsDisk[Disco: generated_reports/]
```

---

## 2. Diagrama de Modelo de Datos

```mermaid
erDiagram
    USUARIO {
        int id PK
        string username
        string role
        string email
    }
    DATASET_MEMBER {
        string image_id PK
        int is_malignant
        string original_class
        float pixel_mean
    }
    MODELO_ENTRENADO {
        string name PK
        string type
        float accuracy
        float auc
        float f1_score
        float mcc
        string saved_path
    }
    PRUEBA_ESTADISTICA {
        int id PK
        string model_a_fk
        string model_b_fk
        string test_name
        float p_value
        string interpretation
    }
    
    MODELO_ENTRENADO ||--o{ PRUEBA_ESTADISTICA : evaluado_en
    DATASET_MEMBER ||--o{ MODELO_ENTRENADO : usado_para_entrenar
```

---

## 3. Diagrama de Componentes

```mermaid
componentDiagram
    package "Frontend Component Stack (Next.js)" {
        [Dashboard Page] --> [Navbar Component]
        [Dashboard Page] --> [Tabs Navigation]
        [Dashboard Page] --> [VoiceChatbot Component]
        [Navbar Component] --> [LanguageSelector]
        [Navbar Component] --> [ThemeToggle]
    }
    
    package "Backend Service Stack (FastAPI)" {
        [FastAPI Core] --> [Auth Router]
        [FastAPI Core] --> [EDA Router]
        [FastAPI Core] --> [Training Router]
        [FastAPI Core] --> [Stats Router]
        [FastAPI Core] --> [Reports Router]
        
        [Training Router] --> [Models Builder]
        [Stats Router] --> [Stats Engine]
        [Reports Router] --> [ReportLab / python-docx / openpyxl]
    }
```

---

## 4. Diagrama de Secuencia (Flujo de Entrenamiento y Pruebas Estadísticas)

```mermaid
sequenceDiagram
    autonumber
    actor Usuario
    participant Frontend as Next.js UI
    participant API as FastAPI Backend
    participant ML as ML & Stats Engine
    participant Disk as Disco (.keras / PDF)

    Usuario->>Frontend: Selecciona pestaña "Entrenamiento"
    Frontend->>API: POST /api/training/run
    API->>ML: Construye 3 clásicos + 2 híbridos
    ML->>Disk: Guarda el mejor modelo como .keras
    ML-->>API: Retorna resumen de Accuracy, AUC, MCC
    API-->>Frontend: Muestra tabla comparativa
    
    Usuario->>Frontend: Selecciona pestaña "Pruebas Estadísticas"
    Frontend->>API: POST /api/statistical/run-all
    API->>ML: Ejecuta KS, Mann-Whitney, Morgan-Pitman, McNemar, LM
    ML-->>API: Retorna resultados con p-valores e interpretaciones
    API-->>Frontend: Despliega tabla de pruebas no paramétricas
```

---

## 5. Diagrama de Estados del Modelo de IA

```mermaid
stateDiagram-v2
    [*] --> NoIniciado: Cargar datos
    NoIniciado --> Preprocesamiento: Iniciar pipeline
    Preprocesamiento --> Entrenamiento: Generadores con Data Augmentation
    
    state Entrenamiento {
        [*] --> EntrenandoClasicos: ResNet, EfficientNet, MobileNet
        EntrenandoClasicos --> EntrenandoHibridos: DenseNet+Attention, EfficientNet+Spatial
    }
    
    Entrenamiento --> Evaluacion5Fold: Finaliza época
    Evaluacion5Fold --> PruebasEstadisticas: Métricas validadas
    PruebasEstadisticas --> GuardadoMejorModelo: p-valores calculados (KS, MW)
    GuardadoMejorModelo --> ReporteGenerado: Exportar PDF/Word/Excel
    ReporteGenerado --> [*]
```
