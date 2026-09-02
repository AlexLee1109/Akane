export type DemoMode = "live" | "showcase";

export interface AkaneWebConfig {
  basePath: string;
  apiUrl: string | null;
  demoMode: DemoMode;
  isProduction: boolean;
  githubUrl: string;
  modelName: string;
}

function normalizeBasePath(value: string | undefined) {
  const trimmed = (value || "/").trim();
  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return `${withLeadingSlash.replace(/\/+/g, "/").replace(/\/+$/, "")}/`.replace(/^\/\/$/, "/");
}

function normalizeApiUrl(value: string | undefined, isProduction: boolean) {
  const trimmed = value?.trim();
  if (!trimmed) return isProduction ? null : window.location.origin;

  try {
    if (isProduction && trimmed.startsWith("/")) {
      throw new Error("VITE_AKANE_API_URL must be an absolute HTTPS URL in production.");
    }
    const url = trimmed.startsWith("/") ? new URL(trimmed, window.location.origin) : new URL(trimmed);
    const isHttp = url.protocol === "http:";
    const isHttps = url.protocol === "https:";
    const isDevelopmentLoopback =
      !isProduction && isHttp && (url.hostname === "localhost" || url.hostname === "127.0.0.1");

    if ((!isHttps && !isDevelopmentLoopback) || url.username || url.password || url.search || url.hash) {
      throw new Error(
        "VITE_AKANE_API_URL must be HTTPS, or loopback HTTP during development, without credentials, query, or fragment.",
      );
    }
    return url.toString().replace(/\/+$/, "");
  } catch (error) {
    if (!isProduction) throw error;
    return null;
  }
}

function readDemoMode(value: string | undefined): DemoMode {
  if (!value || value.trim().toLowerCase() === "live") return "live";
  if (value.trim().toLowerCase() === "showcase") return "showcase";
  if (!import.meta.env.PROD) {
    throw new Error("VITE_DEMO_MODE must be either 'live' or 'showcase'.");
  }
  return "showcase";
}

const isProduction = import.meta.env.PROD;

export const projectConfig: AkaneWebConfig = Object.freeze({
  basePath: normalizeBasePath(import.meta.env.VITE_BASE_PATH || import.meta.env.BASE_URL),
  apiUrl: normalizeApiUrl(import.meta.env.VITE_AKANE_API_URL, isProduction),
  demoMode: readDemoMode(import.meta.env.VITE_DEMO_MODE),
  isProduction,
  githubUrl: import.meta.env.VITE_GITHUB_URL?.trim() || "https://github.com/AlexLee1109/Akane",
  modelName: import.meta.env.VITE_MODEL_DISPLAY_NAME?.trim() || "Qwen3.5 9B",
});
