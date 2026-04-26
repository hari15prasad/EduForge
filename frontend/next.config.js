/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // Required for static export to 'out' directory
  images: {
    unoptimized: true, // Required for static export
  },
  // Note: rewrites are only used for local development 'npm run dev'
  // In production (Hugging Face), FastAPI will serve the static files
};

module.exports = nextConfig;