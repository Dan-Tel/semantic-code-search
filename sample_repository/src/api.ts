export interface ApiRequest {
  headers: Record<string, string>;
  url: string;
}

export function addAuthToken(
  request: ApiRequest,
  token: string
): ApiRequest {
  return {
    ...request,
    headers: {
      ...request.headers,
      Authorization: `Bearer ${token}`,
    },
  };
}

export const removeAuthToken = (
  request: ApiRequest
): ApiRequest => {
  const headers = { ...request.headers };

  delete headers.Authorization;

  return {
    ...request,
    headers,
  };
};

export function createApiUrl(
  baseUrl: string,
  endpoint: string
): string {
  return `${baseUrl}/${endpoint}`;
}