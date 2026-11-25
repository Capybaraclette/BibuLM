import React from 'react';
import bibulmLogo from '../assets/bibulm-logo.png';

interface BibuLMLogoProps {
  className?: string;
}

/**
 * Calico cat logo emblem for BibuLM.
 */
export const BibuLMLogo: React.FC<BibuLMLogoProps> = ({ className = 'w-5 h-5' }) => (
  <img
    src={bibulmLogo}
    alt="BibuLM"
    draggable={false}
    className={`${className} rounded-md object-contain shrink-0 select-none`}
  />
);
