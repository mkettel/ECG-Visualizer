"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Define theme type
type Theme = 'light' | 'dark';

// Define context type
type ThemeContextType = {
  theme: Theme;
  toggleTheme: () => void;
};

// Create the theme context
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Props for the ThemeProvider component
interface ThemeProviderProps {
  children: ReactNode;
}

// Theme provider component
export function ThemeProvider({ children }: ThemeProviderProps) {
  // Use state to track the current theme
  const [theme, setTheme] = useState<Theme>('light');
  const [mounted, setMounted] = useState(false);

  // Initialize theme from localStorage on component mount
  useEffect(() => {
    setMounted(true);
    
    // Get saved theme from localStorage or use system preference
    const savedTheme = localStorage.getItem('theme') as Theme | null;
    
    if (savedTheme) {
      setTheme(savedTheme);
    } else {
      // Use system preference as fallback
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setTheme(systemPrefersDark ? 'dark' : 'light');
    }
  }, []);

  // Apply theme changes to document
  useEffect(() => {
    if (!mounted) return;

    // Save to localStorage
    localStorage.setItem('theme', theme);
    
    // Apply theme class to document
    if (theme === 'dark') {
      document.documentElement.classList.add('dark-theme');
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light-theme');
    } else {
      document.documentElement.classList.add('light-theme');
      document.documentElement.classList.remove('dark-theme');
      document.documentElement.classList.remove('dark');
    }
    
    // Update CSS variables for the theme
    applyThemeColors(theme);
  }, [theme, mounted]);

  // Apply theme colors based on the theme
  const applyThemeColors = (currentTheme: Theme) => {
    const root = document.documentElement;
    
    if (currentTheme === 'dark') {
      // Dark theme colors
      root.style.setProperty('--background', '#171717');
      root.style.setProperty('--foreground', '#f9fafb');
      root.style.setProperty('--chart-bg', '#262626');
      root.style.setProperty('--chart-text', '#f5f5f5');
      root.style.setProperty('--control-bg', '#f5f5f5');
      root.style.setProperty('--control-border', '#525252');
      root.style.setProperty('--accent', 'rgb(239, 68, 68)');
      root.style.setProperty('--accent-light', 'rgba(239, 68, 68, 0.2)');
      root.style.setProperty('--grid-line', '#4B5563');
      root.style.setProperty('--label-color', '#f5f5f5');
      root.style.setProperty('--tick-color', '#9CA3AF');
    } else {
      // Light theme colors
      root.style.setProperty('--background', '#f9fafb');
      root.style.setProperty('--foreground', '#171717');
      root.style.setProperty('--chart-bg', '#ffffff');
      root.style.setProperty('--chart-text', '#1f2937');
      root.style.setProperty('--control-bg', '#f1f5f9');
      root.style.setProperty('--control-border', '#e2e8f0');
      root.style.setProperty('--accent', 'rgb(220, 38, 38)');
      root.style.setProperty('--accent-light', 'rgba(220, 38, 38, 0.1)');
      root.style.setProperty('--grid-line', '#e5e7eb');
      root.style.setProperty('--label-color', '#4b5563');
      root.style.setProperty('--tick-color', '#171717');
    }
  };

  // Toggle theme function
  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // Context value
  const contextValue: ThemeContextType = {
    theme,
    toggleTheme,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
}

// Custom hook to use the theme context
export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// Hook to enable smooth transitions
export function useThemeTransition() {
  useEffect(() => {
    // Add transition to root element
    const root = document.documentElement;
    root.style.setProperty('transition', 'background-color 0.5s ease, color 0.5s ease');
    
    // Add transition styles to various elements
    const style = document.createElement('style');
    style.textContent = `
      * {
        transition: background-color 0.5s ease, color 0.5s ease, border-color 0.5s ease, box-shadow 0.5s ease;
      }
      
      /* Prevent transition on page load */
      .no-transition * {
        transition: none !important;
      }
    `;
    document.head.appendChild(style);
    
    // Remove no-transition class after page load
    setTimeout(() => {
      document.body.classList.remove('no-transition');
    }, 100);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);
}