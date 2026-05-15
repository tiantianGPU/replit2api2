/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PROXY_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
