/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production';

const nextConfig = {
  // Static export only for production HF Space Docker build.
  // Local dev (npm run dev) uses Next.js dev server + rewrites below.
  ...(isProd ? { output: 'export' } : {}),
  images: {
    unoptimized: true,
  },
  // In local dev, proxy /api/* → FastAPI backend on port 8000
  async rewrites() {
    if (isProd) return [];
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:7860/:path*',
      },
    ];
  },
};

module.exports = nextConfig;