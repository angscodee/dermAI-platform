// frontend/src/lib/i18n.ts

export type Language = 'es' | 'en';

export interface TranslationDictionary {
  title: string;
  subtitle: string;
  login_title: string;
  username: string;
  password: string;
  login_btn: string;
  invalid_login: string;
  logout: string;
  tab_diagnosis: string;
  tab_3d: string;
  tab_eda: string;
  tab_training: string;
  tab_cv: string;
  tab_tuning: string;
  tab_stats: string;
  tab_reports: string;
  upload_title: string;
  upload_drop: string;
  select_model: string;
  analyze_btn: string;
  chatbot_title: string;
  chatbot_placeholder: string;
  speak_btn: string;
  send_btn: string;
  clear_chat: string;
  welcome_msg: string;
}

export const translations: Record<Language, TranslationDictionary> = {
  es: {
    title: "DermAI Platform",
    subtitle: "Diagnostico por Redes Neuronales Clasicas e Hibridas, Pruebas Estadisticas Robustas y Reportes",
    login_title: "Autenticacion de Usuario",
    username: "Usuario",
    password: "Password",
    login_btn: "Ingresar al Sistema",
    invalid_login: "Credenciales incorrectas (Usa admin / admin123)",
    logout: "Cerrar Sesion",
    tab_diagnosis: "Diagnostico por Imagen",
    tab_3d: "Visor 3D Total Body",
    tab_eda: "1. EDA (Descriptivo)",
    tab_training: "2. Entrenamiento (3 Clasicos + 2 Hibridos)",
    tab_cv: "3. Validacion Cruzada (5-Fold)",
    tab_tuning: "4. Hiperparametros",
    tab_stats: "5. Pruebas Estadisticas",
    tab_reports: "6. Reportes (PDF, Word, Excel)",
    upload_title: "Cargar Imagen Dermatoscopica / Lesion",
    upload_drop: "Arrastra y suelta una imagen o haz clic para seleccionar",
    select_model: "Seleccionar Modelo de Inferencia",
    analyze_btn: "Realizar Diagnostico IA",
    chatbot_title: "Asistente Virtual (Voz y Texto)",
    chatbot_placeholder: "Escribe tu pregunta o habla...",
    speak_btn: "Hablar",
    send_btn: "Enviar",
    clear_chat: "Limpiar Chat",
    welcome_msg: "Hola, soy el asistente virtual de DermAI. ¿En que puedo ayudarte con el diagnostico o el analisis estadistico?"
  },
  en: {
    title: "DermAI Platform",
    subtitle: "Diagnosis via Classic and Hybrid Neural Networks, Robust Statistical Tests and Reports",
    login_title: "User Authentication",
    username: "Username",
    password: "Password",
    login_btn: "Sign In",
    invalid_login: "Invalid credentials (Use admin / admin123)",
    logout: "Log Out",
    tab_diagnosis: "Image Diagnosis",
    tab_3d: "3D Total Body Viewer",
    tab_eda: "1. EDA (Descriptive)",
    tab_training: "2. Training (3 Classic + 2 Hybrid)",
    tab_cv: "3. Cross Validation (5-Fold)",
    tab_tuning: "4. Hyperparameters",
    tab_stats: "5. Statistical Tests",
    tab_reports: "6. Reports (PDF, Word, Excel)",
    upload_title: "Upload Dermoscopic Image / Lesion",
    upload_drop: "Drag & drop an image or click to browse",
    select_model: "Select Inference Model",
    analyze_btn: "Run AI Diagnosis",
    chatbot_title: "Virtual Assistant (Voice & Text)",
    chatbot_placeholder: "Type your question or speak...",
    speak_btn: "Speak",
    send_btn: "Send",
    clear_chat: "Clear Chat",
    welcome_msg: "Hello, I am the DermAI virtual assistant. How can I assist you with diagnosis or statistical analysis?"
  }
};
