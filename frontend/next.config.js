/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy /api/* → FastAPI backend (uvicorn on :7860)
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:7860/:path*',
      },
    ];
  },
  // Suppress the lockfile workspace root warning
  experimental: {},
};

module.exports = nextConfig;