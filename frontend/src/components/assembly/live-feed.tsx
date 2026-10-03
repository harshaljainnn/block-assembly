import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CheckCircle2,
  Copy,
  Expand,
  ExternalLink,
  Layers,
  Maximize2,
  Minimize2,
  Radio,
  RefreshCw,
  Settings,
  Tv,
  Wifi,
  WifiOff,
} from "lucide-react";
import { cn } from "@/lib/utils";

const DEFAULT_BACKEND_URL =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ||
  (typeof window !== "undefined" && window.location.hostname
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000");
const DEFAULT_RAW_CAMERA_URL = "http://192.168.2.217:4747/video";

export function LiveFeed({ className }: { className?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  const [streamMode, setStreamMode] = useState<"hud" | "raw">("hud");
  const [backendUrl, setBackendUrl] = useState<string>(() => {
    if (typeof window !== "undefined") {
      return (
        localStorage.getItem("block_assembly_backend_url") ||
        DEFAULT_BACKEND_URL
      );
    }
    return DEFAULT_BACKEND_URL;
  });
  const [rawCameraUrl, setRawCameraUrl] = useState<string>(() => {
    if (typeof window !== "undefined") {
      return (
        localStorage.getItem("droidcam_stream_url") || DEFAULT_RAW_CAMERA_URL
      );
    }
    return DEFAULT_RAW_CAMERA_URL;
  });

  const [streamStatus, setStreamStatus] = useState<
    "streaming" | "waiting" | "offline"
  >("waiting");
  const [hudStreamActive, setHudStreamActive] = useState<boolean>(false);
  const [streamKey, setStreamKey] = useState<number>(() => Date.now());
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [copiedCmd, setCopiedCmd] = useState<boolean>(false);

  // Settings form input states
  const [inputBackendUrl, setInputBackendUrl] = useState<string>(backendUrl);
  const [inputRawUrl, setInputRawUrl] = useState<string>(rawCameraUrl);

  const checkBackendHealth = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/health`, {
        signal: AbortSignal.timeout(2500),
      });
      if (res.ok) {
        const data = (await res.json()) as { hud_stream_active?: boolean };
        const active = Boolean(data.hud_stream_active);
        setHudStreamActive(active);

        if (streamMode === "hud") {
          setStreamStatus(active ? "streaming" : "waiting");
        }
      } else {
        setHudStreamActive(false);
        if (streamMode === "hud") setStreamStatus("offline");
      }
    } catch {
      setHudStreamActive(false);
      if (streamMode === "hud") setStreamStatus("offline");
    }
  }, [backendUrl, streamMode]);

  // Periodic health check every 3 seconds
  useEffect(() => {
    void checkBackendHealth();
    const timer = window.setInterval(() => {
      void checkBackendHealth();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [checkBackendHealth]);

  // Sync fullscreen state
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    if (!containerRef.current) return;
    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error("Fullscreen error:", err);
    }
  };

  const reloadStream = () => {
    setStreamKey(Date.now());
    void checkBackendHealth();
  };

  const handleSaveSettings = () => {
    const cleanBackend = inputBackendUrl.trim().replace(/\/+$/, "");
    const cleanRaw = inputRawUrl.trim();
    setBackendUrl(cleanBackend);
    setRawCameraUrl(cleanRaw);
    localStorage.setItem("block_assembly_backend_url", cleanBackend);
    localStorage.setItem("droidcam_stream_url", cleanRaw);
    setShowSettings(false);
    reloadStream();
  };

  const handleResetSettings = () => {
    setInputBackendUrl(DEFAULT_BACKEND_URL);
    setInputRawUrl(DEFAULT_RAW_CAMERA_URL);
    setBackendUrl(DEFAULT_BACKEND_URL);
    setRawCameraUrl(DEFAULT_RAW_CAMERA_URL);
    localStorage.removeItem("block_assembly_backend_url");
    localStorage.removeItem("droidcam_stream_url");
    setShowSettings(false);
    reloadStream();
  };

  const runCommand = `python live_demo.py --camera phone --dashboard ${backendUrl}`;

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(runCommand);
      setCopiedCmd(true);
      setTimeout(() => setCopiedCmd(false), 2000);
    } catch (e) {
      console.error("Clipboard copy failed:", e);
    }
  };

  const currentStreamUrl =
    streamMode === "hud"
      ? `${backendUrl}/api/camera/hud_stream?t=${streamKey}`
      : `${rawCameraUrl}${rawCameraUrl.includes("?") ? "&" : "?"}t=${streamKey}`;

  return (
    <div
      ref={containerRef}
      className={cn(
        "rounded-[10px] border border-border bg-card shadow-sm overflow-hidden transition-all",
        isFullscreen && "fixed inset-0 z-50 rounded-none border-none p-4 bg-background",
        className,
      )}
    >
      {/* Top Header & Stream Controls */}
      <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between border-b border-border bg-card">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
            <Radio className="size-4 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-foreground sm:text-base">
                Live AI Inspection Stream
              </h2>

              {/* Status Badge */}
              {streamMode === "hud" ? (
                streamStatus === "streaming" ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
                    <span className="size-1.5 rounded-full bg-emerald-500 animate-ping" />
                    BOUNDING BOXES ACTIVE
                  </span>
                ) : streamStatus === "waiting" ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-amber-600 dark:text-amber-400">
                    <span className="size-1.5 rounded-full bg-amber-500" />
                    WAITING FOR DETECTOR
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-destructive/30 bg-destructive-soft px-2.5 py-0.5 text-[11px] font-semibold text-destructive">
                    <span className="size-1.5 rounded-full bg-destructive" />
                    FEED OFFLINE
                  </span>
                )
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-sky-600 dark:text-sky-400">
                  <span className="size-1.5 rounded-full bg-sky-500" />
                  RAW CAMERA
                </span>
              )}
            </div>

            <p className="mt-0.5 text-xs text-muted-foreground">
              {streamMode === "hud"
                ? "YOLO11 Block Detection • Dual-Perception AI HUD"
                : `Direct Video Source (${rawCameraUrl})`}
            </p>
          </div>
        </div>

        {/* Action Controls & Feed Switcher */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Feed Switcher Buttons */}
          <div className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => {
                setStreamMode("hud");
                reloadStream();
              }}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 font-medium transition cursor-pointer",
                streamMode === "hud"
                  ? "bg-primary text-primary-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Layers className="size-3.5" />
              AI HUD (Boxes)
            </button>
            <button
              type="button"
              onClick={() => {
                setStreamMode("raw");
                reloadStream();
              }}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 font-medium transition cursor-pointer",
                streamMode === "raw"
                  ? "bg-primary text-primary-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Camera className="size-3.5" />
              Direct Camera
            </button>
          </div>

          {/* Refresh button */}
          <button
            type="button"
            onClick={reloadStream}
            title="Refresh Stream"
            className="flex size-8 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground hover:bg-accent hover:text-foreground transition cursor-pointer"
          >
            <RefreshCw className="size-3.5" />
          </button>

          {/* Settings button */}
          <button
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            title="Stream Settings"
            className={cn(
              "flex size-8 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground hover:bg-accent hover:text-foreground transition cursor-pointer",
              showSettings && "border-primary text-primary bg-primary/10",
            )}
          >
            <Settings className="size-3.5" />
          </button>

          {/* Fullscreen button */}
          <button
            type="button"
            onClick={toggleFullscreen}
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Stream"}
            className="flex size-8 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground hover:bg-accent hover:text-foreground transition cursor-pointer"
          >
            {isFullscreen ? (
              <Minimize2 className="size-3.5" />
            ) : (
              <Maximize2 className="size-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Stream Configuration Panel (Collapsible) */}
      {showSettings && (
        <div className="border-b border-border bg-muted/20 p-4 text-xs transition-all">
          <div className="font-semibold text-foreground mb-2">
            Camera & Telemetry Stream Configuration
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label className="text-muted-foreground block mb-1">
                FastAPI Backend URL (HUD Stream Endpoint)
              </label>
              <input
                type="text"
                value={inputBackendUrl}
                onChange={(e) => setInputBackendUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="w-full rounded-md border border-border bg-card px-2.5 py-1.5 text-xs text-foreground outline-none focus:border-primary font-mono"
              />
              <span className="text-[10px] text-muted-foreground mt-0.5 block">
                Default: <code>http://localhost:8000</code>
              </span>
            </div>

            <div>
              <label className="text-muted-foreground block mb-1">
                Raw DroidCam / IP Stream URL
              </label>
              <input
                type="text"
                value={inputRawUrl}
                onChange={(e) => setInputRawUrl(e.target.value)}
                placeholder="http://192.168.2.217:4747/video"
                className="w-full rounded-md border border-border bg-card px-2.5 py-1.5 text-xs text-foreground outline-none focus:border-primary font-mono"
              />
              <span className="text-[10px] text-muted-foreground mt-0.5 block">
                Direct phone stream for fallback
              </span>
            </div>
          </div>

          <div className="mt-3 flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={handleResetSettings}
              className="rounded-md border border-border bg-card px-3 py-1 text-xs text-muted-foreground hover:text-foreground transition cursor-pointer"
            >
              Reset Defaults
            </button>
            <button
              type="button"
              onClick={handleSaveSettings}
              className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:opacity-90 transition cursor-pointer"
            >
              Apply Settings
            </button>
          </div>
        </div>
      )}

      {/* Video Stream Stage */}
      <div className="relative flex min-h-[360px] max-h-[580px] w-full items-center justify-center overflow-hidden bg-black select-none">
        {/* Live MJPEG Image */}
        <img
          key={`${streamMode}-${streamKey}`}
          src={currentStreamUrl}
          alt="AI Inspection Live Stream"
          onLoad={() => {
            setStreamStatus("streaming");
          }}
          onError={() => {
            if (streamMode === "hud") {
              if (!hudStreamActive) {
                setStreamStatus("waiting");
              }
            } else {
              setStreamStatus("offline");
            }
          }}
          className={cn(
            "h-auto max-h-[560px] w-full object-contain transition-opacity duration-300",
            (streamMode === "hud" && !hudStreamActive) || streamStatus === "offline"
              ? "opacity-25"
              : "opacity-100",
          )}
        />

        {/* Real-time Indicator Inset */}
        {((streamMode === "hud" && hudStreamActive) || streamStatus === "streaming") && (
          <div className="absolute top-3 left-3 flex items-center gap-2 rounded-md bg-black/60 px-2.5 py-1 text-[11px] font-mono text-white backdrop-blur-md border border-white/10">
            <span className="size-2 rounded-full bg-emerald-500 animate-ping" />
            <span>
              {streamMode === "hud"
                ? "AI REAL-TIME INFERENCE"
                : "CAMERA DIRECT FEED"}
            </span>
          </div>
        )}

        {/* Fallback Overlay when CV Detector is not streaming */}
        {((streamMode === "hud" && !hudStreamActive) || streamStatus === "offline") && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-black/75 backdrop-blur-xs">
            <div className="max-w-md rounded-xl border border-zinc-800 bg-zinc-900/90 p-6 text-zinc-100 shadow-2xl">
              <div className="mx-auto mb-3 flex size-12 items-center justify-center rounded-xl border border-amber-500/20 bg-amber-500/10 text-amber-400">
                <Tv className="size-6" />
              </div>

              <h3 className="text-base font-bold text-white">
                {streamMode === "hud"
                  ? "AI HUD Stream Waiting for CV Detector"
                  : "Camera Feed Not Available"}
              </h3>

              <p className="mt-2 text-xs text-zinc-400">
                {streamMode === "hud"
                  ? "Start the computer vision inspection pipeline in your terminal to stream the real-time AI bounding boxes and state HUD:"
                  : `Could not connect to direct camera at ${rawCameraUrl}. Check your camera WiFi or IP address.`}
              </p>

              {streamMode === "hud" && (
                <div className="mt-3.5 flex items-center justify-between rounded-lg border border-zinc-800 bg-black/80 px-3 py-2 text-[11px] font-mono text-emerald-400 select-all">
                  <span className="truncate pr-2">{runCommand}</span>
                  <button
                    type="button"
                    onClick={copyCommand}
                    title="Copy command"
                    className="flex items-center gap-1 rounded bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-200 hover:bg-zinc-700 transition cursor-pointer"
                  >
                    {copiedCmd ? (
                      <>
                        <CheckCircle2 className="size-3 text-emerald-400" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="size-3" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
              )}

              <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
                {streamMode === "hud" ? (
                  <button
                    type="button"
                    onClick={() => {
                      setStreamMode("raw");
                      reloadStream();
                    }}
                    className="rounded-lg border border-zinc-700 bg-zinc-800 px-3.5 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-zinc-700 transition cursor-pointer"
                  >
                    View Raw Camera Feed
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      setStreamMode("hud");
                      reloadStream();
                    }}
                    className="rounded-lg border border-zinc-700 bg-zinc-800 px-3.5 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-zinc-700 transition cursor-pointer"
                  >
                    Switch to AI HUD Feed
                  </button>
                )}

                <button
                  type="button"
                  onClick={reloadStream}
                  className="rounded-lg bg-emerald-600 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500 transition cursor-pointer"
                >
                  Retry Connection
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Info Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border bg-card px-4 py-2 text-[11px] text-muted-foreground">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-foreground">Backend:</span>
          <span className="font-mono">{backendUrl}</span>
          <span className="text-border">•</span>
          <span className="font-semibold text-foreground">Stream:</span>
          <span>{streamMode === "hud" ? "/api/camera/hud_stream" : rawCameraUrl}</span>
        </div>

        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span
              className={cn(
                "size-2 rounded-full",
                hudStreamActive ? "bg-emerald-500" : "bg-zinc-400",
              )}
            />
            Detector: {hudStreamActive ? "Active" : "Standby"}
          </span>
        </div>
      </div>
    </div>
  );
}
