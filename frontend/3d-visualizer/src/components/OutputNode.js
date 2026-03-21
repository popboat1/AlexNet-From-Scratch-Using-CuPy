import React from 'react';
import { Edges } from '@react-three/drei';

export const OutputNode = ({ position, isSelected }) => {
    return (
        <group position={position}>
            <mesh>
                <boxGeometry args={[0.5, 0.5, 0.5]} />
                <meshBasicMaterial color="#1a1005" transparent opacity={0.8} />
                <Edges scale={1.05} threshold={15} color={isSelected ? "#ffaa00" : "#ff5500"} />
            </mesh>
            <mesh position={[0, 0.6, 0]}>
                <planeGeometry args={[1, 0.3]} />
                <meshBasicMaterial color="black" transparent opacity={0.5} />
            </mesh>
        </group>
    );
};