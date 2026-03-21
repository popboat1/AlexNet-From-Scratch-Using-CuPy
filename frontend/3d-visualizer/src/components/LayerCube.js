import React, { useState, useMemo, useEffect } from 'react';
import { Edges } from '@react-three/drei';
import * as THREE from 'three';

const FeatureSlice = ({ texture, gridSize, index, sizeY, sizeZ, xOffset, geometry }) => {
    const sliceTexture = useMemo(() => {
        const cloned = texture.clone();
        cloned.magFilter = THREE.NearestFilter;
        cloned.minFilter = THREE.NearestFilter;
        cloned.repeat.set(1 / gridSize, 1 / gridSize);
        const col = index % gridSize;
        const row = Math.floor(index / gridSize);
        cloned.offset.set(col / gridSize, 1 - (row + 1) / gridSize);
        
        if (cloned.image) {
            cloned.needsUpdate = true;
        }
        
        return cloned;
    }, [texture, gridSize, index]);

    useEffect(() => {
        return () => {
            sliceTexture.dispose();
        };
    }, [sliceTexture]);

    return (
        <mesh position={[xOffset, 0, 0]} rotation={[0, -Math.PI / 2, 0]} geometry={geometry}>
            <meshBasicMaterial
                map={sliceTexture}
                transparent={true}
                opacity={0.8}
                alphaTest={0.1}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
                side={THREE.DoubleSide}
            />
        </mesh>
    );
};

export const LayerCube = ({ position, size, shape, textureUrl, isSelected }) => {
    const [sizeX, sizeY, sizeZ] = size;
    const channels = shape[2] || 1;
    const numFeatures = Math.min(channels, 24); 
    const gridSize = Math.ceil(Math.sqrt(Math.min(channels, 64)));

    const [imageEl, setImageEl] = useState(null);

    useEffect(() => {
        if (!textureUrl) return;
        const img = new window.Image();
        img.src = textureUrl;
        img.onload = () => setImageEl(img);
    }, [textureUrl]);

    const sliceGeometry = useMemo(() => new THREE.PlaneGeometry(sizeZ, sizeY), [sizeZ, sizeY]);
    const fallbackTexture = useMemo(() => new THREE.Texture(), []);

    const baseTexture = useMemo(() => {
        if (!imageEl) return fallbackTexture;
        const tex = new THREE.Texture(imageEl);
        tex.needsUpdate = true;
        return tex;
    }, [imageEl, fallbackTexture]);

    return (
        <group position={position}>
            <mesh>
                <boxGeometry args={[sizeX, sizeY, sizeZ]} />
                <meshBasicMaterial color="#050a1f" transparent opacity={0.3} side={THREE.DoubleSide} depthWrite={false} />
                <Edges scale={1.01} threshold={15} color={isSelected ? "#00ffcc" : "#4a90e2"} />
            </mesh>

            {Array.from({ length: numFeatures }).map((_, index) => {
                const xOffset = (-sizeX / 2) * 0.95 + (sizeX * 0.95 * index) / Math.max(1, numFeatures - 1);
                return (
                    <FeatureSlice
                        key={index}
                        texture={baseTexture}
                        gridSize={gridSize}
                        index={index}
                        sizeY={sizeY}
                        sizeZ={sizeZ}
                        xOffset={xOffset}
                        geometry={sliceGeometry}
                    />
                );
            })}
        </group>
    );
};