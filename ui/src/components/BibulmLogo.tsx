import React from 'react';
import bibulmLogo from '../assets/bibulm-logo.png';

interface BibulmLogoProps {
  className?: string;
}

/**
 * Calico cat logo emblem for Bibulm.
 */
export const BibulmLogo: React.FC<BibulmLogoProps> = ({ className = 'w-5 h-5' }) => (
  <img
    src={bibulmLogo}
    alt="Bibulm"
    draggable={false}
    className={`${className} rounded-md object-contain shrink-0 select-none`}
  />
);
