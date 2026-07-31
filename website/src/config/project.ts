const searchParams = new URLSearchParams(window.location.search);

export const projectConfig = {
  githubUrl: import.meta.env.VITE_GITHUB_URL ?? "https://github.com/AlexLee1109/Akane",
  apiUrl: import.meta.env.VITE_AKANE_API_URL ?? "/",
  apiToken: searchParams.get("api_token")?.trim() ?? "",
  showcaseUrl: import.meta.env.VITE_SHOWCASE_URL ?? "",
  modelName: import.meta.env.VITE_MODEL_DISPLAY_NAME ?? "Gemma 4 E4B",
};
