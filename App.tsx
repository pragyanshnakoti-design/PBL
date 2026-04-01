import { useState, useEffect } from 'react';
import { LandingPage } from '@/sections/LandingPage';
import { UserDashboard } from '@/sections/UserDashboard';
import { AdminDashboard } from '@/sections/AdminDashboard';
import { useAuth } from '@/hooks/useAuth';
import { useData } from '@/hooks/useData';
import { Toaster } from '@/components/ui/sonner';
import { toast } from 'sonner';
import type { ServiceType, User, RequestStatus } from '@/types';
import './App.css';

function App() {
  const { isAuthenticated, user, isLoading, login, logout, hasRole } = useAuth();
  const { 
    isInitialized, 
    createRequest, 
    getUserRequests, 
    getAllRequests, 
    getUserReports, 
    getAllRooms, 
    updateRequestStatus,
    getDashboardStats 
  } = useData();

  const [userRequests, setUserRequests] = useState<any[]>([]);
  const [allRequests, setAllRequests] = useState<any[]>([]);
  const [userReports, setUserReports] = useState<any[]>([]);
  const [rooms, setRooms] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);

  // Refresh data when needed
  const refreshData = () => {
    if (user) {
      setUserRequests(getUserRequests(user.id));
      setUserReports(getUserReports(user.id));
    }
    setAllRequests(getAllRequests());
    setRooms(getAllRooms());
    setStats(getDashboardStats());
  };

  // Initial data load
  useEffect(() => {
    if (isInitialized) {
      refreshData();
    }
  }, [isInitialized, user]);

  // Handle login
  const handleLogin = async (email: string, password: string) => {
    const result = await login(email, password);
    if (result.success) {
      toast.success('Login successful!', {
        description: 'Welcome to ZetaTech Hospital Management System'
      });
    } else {
      toast.error('Login failed', {
        description: result.error
      });
    }
    return result;
  };

  // Handle logout
  const handleLogout = () => {
    logout();
    toast.info('Logged out successfully');
  };

  // Handle create request
  const handleCreateRequest = (serviceType: ServiceType) => {
    if (user) {
      createRequest(user.id, user.email, serviceType);
      refreshData();
      toast.success('Service request created!', {
        description: `Your ${serviceType} request has been submitted and is pending approval.`
      });
    }
  };

  // Handle update request status
  const handleUpdateRequestStatus = (requestId: string, status: RequestStatus, reason?: string) => {
    updateRequestStatus(requestId, status, reason);
    refreshData();
    toast.success(`Request ${status}!`, {
      description: reason || `The request has been ${status}.`
    });
  };

  // Show loading state
  if (isLoading || !isInitialized) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-4 animate-pulse">
            <svg className="w-8 h-8 text-white animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100">ZetaTech</h2>
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {!isAuthenticated ? (
        <LandingPage onLogin={handleLogin} />
      ) : hasRole('admin') ? (
        <AdminDashboard
          user={user as User}
          requests={allRequests}
          rooms={rooms}
          stats={stats}
          onLogout={handleLogout}
          onUpdateRequestStatus={handleUpdateRequestStatus}
        />
      ) : (
        <UserDashboard
          user={user as User}
          requests={userRequests}
          reports={userReports}
          onLogout={handleLogout}
          onCreateRequest={handleCreateRequest}
        />
      )}
      <Toaster position="top-right" richColors />
    </>
  );
}

export default App;
