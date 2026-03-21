import React, { Suspense } from 'react';
import { Edges, useTexture, useVideoTexture } from '@react-three/drei';
import * as THREE from 'three';

const ImageMaterial = ({ url }) => {
    const texture = useTexture(url);
    return <meshBasicMaterial map={texture} side={THREE.DoubleSide} />;
};

const VideoMaterial = ({ url }) => {
    const texture = useVideoTexture(url);
    return <meshBasicMaterial map={texture} side={THREE.DoubleSide} />;
};

export const InputCube = ({ position, imageUrl, isVideo }) => {
    const sizeX = 0.2;
    const sizeY = 4.5;
    const sizeZ = 4.5;

    return (
        <group position={position}>
            <mesh>
                <boxGeometry args={[sizeX, sizeY, sizeZ]} />
                <meshBasicMaterial color="#111" transparent opacity={0.3} side={THREE.DoubleSide} />
                <Edges scale={1.01} color="#ffffff" />
            </mesh>
            <mesh position={[-sizeX / 2 - 0.01, 0, 0]} rotation={[0, -Math.PI / 2, 0]}>
                <planeGeometry args={[sizeZ * 0.98, sizeY * 0.98]} />
                <Suspense fallback={<meshBasicMaterial color="#222" side={THREE.DoubleSide} />}>
                    {imageUrl ? (
                        isVideo ? <VideoMaterial url={imageUrl} /> : <ImageMaterial url={imageUrl} />
                    ) : (
                        <meshBasicMaterial color="#222" side={THREE.DoubleSide} />
                    )}
                </Suspense>
            </mesh>
        </group>
    );
};