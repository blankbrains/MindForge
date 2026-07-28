const RELOAD_KEY_PREFIX = "mindforge:chunk-reload:";

export async function importWithReload<T>(
  chunkName: string,
  importer: () => Promise<T>,
): Promise<T> {
  const reloadKey = `${RELOAD_KEY_PREFIX}${chunkName}`;
  try {
    const module = await importer();
    sessionStorage.removeItem(reloadKey);
    return module;
  } catch (error) {
    if (!sessionStorage.getItem(reloadKey)) {
      sessionStorage.setItem(reloadKey, "1");
      window.location.reload();
      throw error;
    }
    sessionStorage.removeItem(reloadKey);
    throw error;
  }
}
