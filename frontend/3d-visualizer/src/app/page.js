"use client";

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { InputCube } from '@/components/InputCube';
import { LayerCube } from '@/components/LayerCube';
import { OutputNode } from '@/components/OutputNode';

export default function NetworkVisualizer() {
    const [activeTab, setActiveTab] = useState({ type: 'prediction', layerIndex: null });
    const [layerData, setLayerData] = useState([]);
    const [prediction, setPrediction] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [isVideo, setIsVideo] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [blockGap, setBlockGap] = useState(0.2);
    const [isPanelOpen, setIsPanelOpen] = useState(true);
    const [zoomedFeature, setZoomedFeature] = useState(null);

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const isAwaitingResponse = useRef(false);
    const animationFrameId = useRef(null);
    const lastFrameTime = useRef(0);
    const isVideoStateRef = useRef(false);

    const connectWebSocket = () => {
        if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
            return;
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/predict-video`;

        wsRef.current = new WebSocket(wsUrl);
        
        wsRef.current.onmessage = (event) => {
            if (!isVideoStateRef.current) return; 
            const data = JSON.parse(event.data);
            setLayerData(data.layers);
            setPrediction(data.prediction);
            isAwaitingResponse.current = false; 
        };

        wsRef.current.onclose = () => {
            console.log("WebSocket closed by server.");
            isAwaitingResponse.current = false;
        };
    };

    useEffect(() => {
        connectWebSocket();
        return () => {
            if (wsRef.current) wsRef.current.close();
            cancelAnimationFrame(animationFrameId.current);
        };
    }, []);

    const processVideoFrame = (timestamp) => {
        if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
            animationFrameId.current = requestAnimationFrame(processVideoFrame);
            return;
        }

        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
            connectWebSocket();
        }

        if (timestamp - lastFrameTime.current >= 80) {
            if (!isAwaitingResponse.current && wsRef.current?.readyState === WebSocket.OPEN) {
                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d', { willReadFrequently: true });
                ctx.drawImage(videoRef.current, 0, 0, 227, 227);
                const base64Frame = canvas.toDataURL('image/jpeg', 0.4); 
                isAwaitingResponse.current = true;
                wsRef.current.send(base64Frame);
                lastFrameTime.current = timestamp;
            }
        }
        animationFrameId.current = requestAnimationFrame(processVideoFrame);
    };

    const layout = useMemo(() => {
        const defaultData = [
            { layer_index: 1, shape: [55, 55, 96], texture_b64: null },
            { layer_index: 2, shape: [27, 27, 256], texture_b64: null },
            { layer_index: 3, shape: [13, 13, 384], texture_b64: null },
            { layer_index: 4, shape: [13, 13, 384], texture_b64: null },
            { layer_index: 5, shape: [13, 13, 256], texture_b64: null }
        ];

        const activeData = layerData.length > 0 ? layerData : defaultData;
        let currentX = -6; 

        const inputPos = previewImage ? [currentX, 0, 0] : null;
        if (previewImage) currentX += 0.1 + (blockGap * 0.5); 

        const mappedLayers = activeData.map((layer) => {
            const sizeY = Math.max(0.8, layer.shape[0] * 0.07);
            const sizeZ = Math.max(0.8, layer.shape[1] * 0.07);
            const sizeX = Math.max(0.8, layer.shape[2] * 0.01);
            const xPos = currentX + sizeX / 2;
            currentX = xPos + sizeX / 2 + blockGap;
            return { ...layer, size: [sizeX, sizeY, sizeZ], xPos };
        });

        return { inputPos, mappedLayers, outputPos: [currentX, 0, 0] };
    }, [layerData, previewImage, blockGap]);

    const activeLayerPanelData = activeTab.type === 'layer' 
        ? layerData.find(l => l.layer_index === activeTab.layerIndex) || layout.mappedLayers.find(l => l.layer_index === activeTab.layerIndex)
        : null;

    const tabs = ['Output', ...layout.mappedLayers.map(l => l.layer_index)];
    const currentTabIndex = activeTab.type === 'prediction' ? 0 : tabs.indexOf(activeTab.layerIndex);

    const handlePrevTab = () => {
        const newIdx = currentTabIndex === 0 ? tabs.length - 1 : currentTabIndex - 1;
        const val = tabs[newIdx];
        if (val === 'Output') setActiveTab({ type: 'prediction', layerIndex: null });
        else setActiveTab({ type: 'layer', layerIndex: val });
    };

    const handleNextTab = () => {
        const newIdx = currentTabIndex === tabs.length - 1 ? 0 : currentTabIndex + 1;
        const val = tabs[newIdx];
        if (val === 'Output') setActiveTab({ type: 'prediction', layerIndex: null });
        else setActiveTab({ type: 'layer', layerIndex: val });
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        isAwaitingResponse.current = false;
        if (videoRef.current) {
            videoRef.current.pause();
            videoRef.current.removeAttribute('src');
            videoRef.current.load();
        }

        const fileUrl = URL.createObjectURL(file);
        const isVid = file.type.startsWith('video/');
        
        setIsVideo(isVid);
        isVideoStateRef.current = isVid; 
        setPreviewImage(fileUrl);
        setActiveTab({ type: 'prediction', layerIndex: null });
        setZoomedFeature(null);
        setIsPanelOpen(true); 

        if (isVid) return; 

        setIsProcessing(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            setLayerData(data.layers);
            setPrediction(data.prediction); 
        } catch (error) {
            console.error("Error:", error);
            alert("Make sure your FastAPI server is running!");
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="relative w-screen h-screen bg-[#050505] text-white font-sans overflow-hidden">
            
            <video 
                ref={videoRef} 
                src={isVideo && previewImage ? previewImage : undefined} 
                className="hidden absolute top-0 left-0 w-0 h-0" 
                autoPlay muted loop playsInline
                onPlay={() => {
                    // Start the loop if it isn't already running
                    if (!animationFrameId.current) {
                        animationFrameId.current = requestAnimationFrame(processVideoFrame);
                    }
                }}
            />
            <canvas ref={canvasRef} width="227" height="227" className="hidden absolute top-0 left-0 w-0 h-0" />

            <div className="absolute inset-0 z-0">
                <Canvas 
                    camera={{ position: [20, 8, 5], fov: 45 }}
                    gl={{ antialias: false, powerPreference: "high-performance" }}
                    dpr={[1, 1.5]}
                >
                    <ambientLight intensity={1.5} />
                    <group rotation={[0, Math.PI / 8, 0]}>
                        {layout.inputPos && (
                            <InputCube position={layout.inputPos} imageUrl={previewImage} isVideo={isVideo} />
                        )}
                        {layout.mappedLayers.map((layer) => (
                            <LayerCube
                                key={layer.layer_index}
                                position={[layer.xPos, 0, 0]}
                                size={layer.size}
                                shape={layer.shape}
                                textureUrl={layer.texture_b64}
                                isSelected={activeTab.type === 'layer' && activeTab.layerIndex === layer.layer_index}
                            />
                        ))}
                        {layout.outputPos && (
                            <OutputNode position={layout.outputPos} isSelected={activeTab.type === 'prediction'} />
                        )}
                    </group>
                    <OrbitControls makeDefault enablePan={true} enableZoom={true} target={[0, 0, 0]} />
                </Canvas>

                {isProcessing && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-10">
                        <h2 className="text-2xl md:text-3xl text-[#00ffcc] font-mono font-bold animate-pulse drop-shadow-lg">Processing Inference...</h2>
                    </div>
                )}
            </div>

            <div className="absolute top-6 left-6 right-16 md:right-auto z-20 pointer-events-none">
                <h1 className="text-2xl md:text-3xl font-black tracking-tight text-white drop-shadow-lg">
                    ALEXNET<span className="text-[#00ffcc]"> VISUALIZER</span>
                </h1>
                <p className="text-xs md:text-sm text-gray-300 mt-2 max-w-xs drop-shadow-md leading-relaxed">
                    Real-time Convolutional Neural Network inference. Drop an image or video to extract and render 3D architectural feature maps.
                </p>
            </div>

            <div className="absolute bottom-6 left-6 right-6 md:right-auto z-20 bg-[#0f0f0f]/80 backdrop-blur-xl border border-white/10 p-4 md:p-6 rounded-2xl flex flex-col md:flex-row items-stretch md:items-center gap-4 md:gap-8 shadow-2xl">
                <div className="flex justify-between items-center w-full md:w-auto">
                    <div>
                        <h2 className="text-base md:text-lg font-bold mb-1">Input Source</h2>
                        <label className="cursor-pointer bg-white text-black font-bold border border-white px-4 py-2 rounded-lg inline-block text-xs md:text-sm transition-colors hover:bg-gray-200">
                            <span>Select File</span>
                            <input type="file" className="hidden" accept="image/*,video/*" onChange={handleFileUpload} />
                        </label>
                    </div>
                    {previewImage && (
                        <div className="h-16 w-16 md:hidden border border-white/20 rounded-lg overflow-hidden relative shrink-0">
                            {isVideo && <div className="absolute top-1 right-1 bg-red-500 w-2 h-2 rounded-full animate-pulse z-10"></div>}
                            {isVideo ? <video src={previewImage} className="w-full h-full object-cover" autoPlay muted loop playsInline /> : <img src={previewImage} alt="Input" className="w-full h-full object-cover" />}
                        </div>
                    )}
                </div>

                <div className="w-full md:w-48 border-t md:border-t-0 md:border-l border-white/10 pt-4 md:pt-0 md:pl-6">
                    <h3 className="text-xs md:text-sm font-semibold text-gray-300 mb-2">Architecture Spacing</h3>
                    <input
                        type="range"
                        min="0.2" max="4.0" step="0.1"
                        value={blockGap}
                        onChange={(e) => setBlockGap(parseFloat(e.target.value))}
                        className="w-full accent-[#00ffcc] cursor-ew-resize"
                    />
                </div>

                {previewImage && (
                    <div className="hidden md:block h-16 w-16 border border-white/20 rounded-lg overflow-hidden relative shrink-0 shadow-lg">
                        {isVideo && <div className="absolute top-1 right-1 bg-red-500 w-2 h-2 rounded-full animate-pulse z-10"></div>}
                        {isVideo ? <video src={previewImage} className="w-full h-full object-cover" autoPlay muted loop playsInline /> : <img src={previewImage} alt="Input" className="w-full h-full object-cover" />}
                    </div>
                )}
            </div>

            {!isPanelOpen && (
                <button
                    onClick={() => setIsPanelOpen(true)}
                    className="absolute top-6 right-6 z-30 bg-[#111] border border-white/10 text-[#00ffcc] font-mono font-bold px-4 py-2 rounded-lg shadow-2xl hover:bg-[#222] transition-colors text-xs md:text-sm"
                >
                    ◀ DATA PANEL
                </button>
            )}

            <div 
                className={`absolute top-0 right-0 bottom-0 md:top-6 md:right-6 md:bottom-6 w-full md:w-[400px] z-40 bg-[#0f0f0f]/95 md:bg-[#0f0f0f]/80 backdrop-blur-2xl md:border border-white/10 md:rounded-2xl flex flex-col shadow-2xl overflow-hidden transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${isPanelOpen ? 'translate-x-0' : 'translate-x-full md:translate-x-[120%]'}`}
            >
                <div className="p-4 md:p-5 border-b border-white/10 bg-black/40 flex justify-between items-center">
                    <div className="flex items-center justify-between bg-[#111] border border-gray-700 rounded-lg p-1 w-48 shadow-inner">
                        <button onClick={handlePrevTab} className="px-3 py-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors text-sm">◀</button>
                        <span className="font-mono font-bold text-sm text-[#00ffcc] tracking-widest">
                            {activeTab.type === 'prediction' ? 'OUTPUT' : `LAYER ${activeTab.layerIndex}`}
                        </span>
                        <button onClick={handleNextTab} className="px-3 py-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors text-sm">▶</button>
                    </div>
                    <button 
                        onClick={() => setIsPanelOpen(false)} 
                        className="text-gray-500 hover:text-white font-bold p-2 rounded-full hover:bg-white/10 transition-colors"
                        title="Close Panel"
                    >
                        ✕
                    </button>
                </div>

                <div className="p-6 flex-1 overflow-y-auto relative custom-scrollbar">
                    {activeTab.type === 'layer' && activeLayerPanelData && (
                        <div className="animate-fadeIn">
                            <div className="mb-6">
                                <p className="text-[10px] md:text-xs text-gray-500 uppercase tracking-wider mb-1">Tensor Shape</p>
                                <p className="font-mono text-xs md:text-sm bg-black/50 px-3 py-2 rounded-lg border border-white/10 tracking-widest inline-block text-gray-200">
                                    {activeLayerPanelData.shape?.join(' × ')}
                                </p>
                            </div>
                            <div>
                                <p className="text-[10px] md:text-xs text-gray-500 uppercase tracking-wider mb-2">Activated Feature Maps</p>
                                <p className="text-xs text-gray-400 mb-3">Click a specific feature to enlarge.</p>
                                
                                {(() => {
                                    const channels = activeLayerPanelData.shape?.[2] || 1;
                                    const maxFeatures = Math.min(channels, 64);
                                    const gridSize = Math.ceil(Math.sqrt(maxFeatures));
                                    
                                    return (
                                        <div 
                                            className="grid gap-[2px] border border-white/10 bg-black/50 p-1 rounded-xl min-h-[200px] relative shadow-inner"
                                            style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}
                                        >
                                            {Array.from({ length: maxFeatures }).map((_, i) => {
                                                const col = i % gridSize;
                                                const row = Math.floor(i / gridSize);
                                                const bgPosX = gridSize > 1 ? (col / (gridSize - 1)) * 100 : 0;
                                                const bgPosY = gridSize > 1 ? (row / (gridSize - 1)) * 100 : 0;
                                                const hasData = !!activeLayerPanelData.texture_b64;

                                                return (
                                                    <div 
                                                        key={i}
                                                        onClick={() => hasData && setZoomedFeature({ layer: activeLayerPanelData.layer_index, index: i, bgPosX, bgPosY, gridSize })}
                                                        className="aspect-square bg-black/80 rounded-sm cursor-pointer hover:ring-2 hover:ring-[#00ffcc] hover:z-10 transition-all relative"
                                                        style={{
                                                            backgroundImage: hasData ? `url(${activeLayerPanelData.texture_b64})` : 'none',
                                                            backgroundSize: `${gridSize * 100}% ${gridSize * 100}%`,
                                                            backgroundPosition: `${bgPosX}% ${bgPosY}%`,
                                                            imageRendering: 'pixelated'
                                                        }}
                                                    />
                                                );
                                            })}
                                        </div>
                                    );
                                })()}
                            </div>
                        </div>
                    )}

                    {activeTab.type === 'prediction' && (
                        <div className="flex flex-col items-center justify-center h-full text-center pb-10 animate-fadeIn">
                            <p className="text-xs text-gray-500 uppercase tracking-wider mb-4 font-semibold">Final Classification</p>
                            <div className="bg-black/50 border border-[#f97316]/30 rounded-2xl p-6 md:p-8 w-full shadow-[0_0_30px_rgba(249,115,22,0.1)]">
                                <h1 className="text-2xl md:text-3xl font-black text-[#f97316] uppercase tracking-widest break-words drop-shadow-md">
                                    {prediction || "No Data"}
                                </h1>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {zoomedFeature && (() => {
                const liveLayer = layerData.find(l => l.layer_index === zoomedFeature.layer);
                const liveUrl = liveLayer ? liveLayer.texture_b64 : "";

                return (
                    <div 
                        className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
                        onClick={() => setZoomedFeature(null)}
                    >
                        <div 
                            className="bg-[#0f0f0f] border border-white/10 p-6 md:p-8 rounded-2xl shadow-2xl flex flex-col items-center max-w-full"
                            onClick={(e) => e.stopPropagation()} 
                        >
                            <h3 className="text-[#00ffcc] font-bold font-mono mb-6 text-lg md:text-xl tracking-widest text-center">
                                LAYER {zoomedFeature.layer} <span className="text-gray-600">{"//"}</span> FEATURE {zoomedFeature.index + 1}
                            </h3>
                            <div 
                                className="w-64 h-64 md:w-96 md:h-96 border-2 border-white/10 rounded-xl bg-black shadow-inner"
                                style={{
                                    backgroundImage: `url(${liveUrl})`,
                                    backgroundSize: `${zoomedFeature.gridSize * 100}% ${zoomedFeature.gridSize * 100}%`,
                                    backgroundPosition: `${zoomedFeature.bgPosX}% ${zoomedFeature.bgPosY}%`,
                                    imageRendering: 'pixelated'
                                }}
                            />
                            <button 
                                className="mt-8 w-full bg-white hover:bg-gray-200 text-black font-bold py-3 rounded-lg transition-colors text-sm"
                                onClick={() => setZoomedFeature(null)}
                            >
                                Close View
                            </button>
                        </div>
                    </div>
                );
            })()}
        </div>
    );
}