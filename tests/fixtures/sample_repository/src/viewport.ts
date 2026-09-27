export interface Viewport {
  zoom: number;
  centerX: number;
  centerY: number;
}

export function saveViewportToUrl(
  viewport: Viewport
): void {
  const params = new URLSearchParams({
    zoom: String(viewport.zoom),
    centerX: String(viewport.centerX),
    centerY: String(viewport.centerY),
  });

  window.history.replaceState(
    null,
    "",
    `?${params.toString()}`
  );
}

export const limitZoom = (
  zoom: number,
  minimumZoom: number,
  maximumZoom: number
): number => {
  return Math.min(
    maximumZoom,
    Math.max(minimumZoom, zoom)
  );
};