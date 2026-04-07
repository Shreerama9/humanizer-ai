import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
        has: [{ type: "header", key: "x-proxy-to-api" }],
      },
    ];
  },
};

export default nextConfig;
