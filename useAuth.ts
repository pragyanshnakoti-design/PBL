import { useState, useEffect, useCallback } from 'react';
import type { User, UserRole, AuthState } from '@/types';

// Mock users for demo
const MOCK_USERS: User[] = [
  {
    id: '1',
    email: 'patient@example.com',
    name: 'John Patient',
    role: 'user',
    passwordHash: 'hashed_password123'
  },
  {
    id: '2',
    email: 'admin@zetatech.com',
    name: 'Admin User',
    role: 'admin',
    passwordHash: 'hashed_admin456'
  }
];

// Simple hash function for demo
const simpleHash = (str: string): string => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16);
};

// Generate JWT-like token
const generateToken = (user: User): string => {
  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const payload = btoa(JSON.stringify({
    sub: user.id,
    email: user.email,
    role: user.role,
    iat: Date.now(),
    exp: Date.now() + 24 * 60 * 60 * 1000 // 24 hours
  }));
  const signature = simpleHash(header + payload);
  return `${header}.${payload}.${signature}`;
};

export const useAuth = () => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    token: null
  });
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('zetatech_token');
    const storedUser = localStorage.getItem('zetatech_user');
    
    if (storedToken && storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setAuthState({
          isAuthenticated: true,
          user,
          token: storedToken
        });
      } catch {
        localStorage.removeItem('zetatech_token');
        localStorage.removeItem('zetatech_user');
      }
    }
    setIsLoading(false);
  }, []);

  const login = useCallback(async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));

    const user = MOCK_USERS.find(u => u.email === email);
    
    if (!user) {
      return { success: false, error: 'User not found' };
    }

    // Verify password (in real app, compare hashes)
    const passwordHash = simpleHash(password);
    if (passwordHash !== user.passwordHash && password !== 'password123' && password !== 'admin456') {
      return { success: false, error: 'Invalid password' };
    }

    const token = generateToken(user);
    
    // Store session
    localStorage.setItem('zetatech_token', token);
    localStorage.setItem('zetatech_user', JSON.stringify(user));

    setAuthState({
      isAuthenticated: true,
      user,
      token
    });

    return { success: true };
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('zetatech_token');
    localStorage.removeItem('zetatech_user');
    setAuthState({
      isAuthenticated: false,
      user: null,
      token: null
    });
  }, []);

  const hasRole = useCallback((role: UserRole): boolean => {
    return authState.user?.role === role;
  }, [authState.user]);

  return {
    ...authState,
    isLoading,
    login,
    logout,
    hasRole
  };
};
