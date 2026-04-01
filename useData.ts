import { useState, useEffect, useCallback } from 'react';
import type { ServiceRequest, MedicalReport, IoTRoom, DashboardStats, ServiceType, RequestStatus } from '@/types';

// Generate unique ID
const generateId = () => Math.random().toString(36).substring(2, 15);

// Mock data generators
const createMockRequests = (): ServiceRequest[] => [
  {
    id: generateId(),
    userId: '1',
    userEmail: 'patient@example.com',
    serviceType: 'consultation',
    status: 'approved',
    reason: 'Doctor available on requested date',
    createdAt: new Date(Date.now() - 86400000 * 2).toISOString(),
    updatedAt: new Date(Date.now() - 86400000).toISOString()
  },
  {
    id: generateId(),
    userId: '1',
    userEmail: 'patient@example.com',
    serviceType: 'therapy',
    status: 'pending',
    createdAt: new Date(Date.now() - 3600000).toISOString(),
    updatedAt: new Date(Date.now() - 3600000).toISOString()
  },
  {
    id: generateId(),
    userId: '3',
    userEmail: 'sarah@example.com',
    serviceType: 'followup',
    status: 'rejected',
    reason: 'Please reschedule for next week',
    createdAt: new Date(Date.now() - 172800000).toISOString(),
    updatedAt: new Date(Date.now() - 86400000).toISOString()
  }
];

const createMockReports = (): MedicalReport[] => [
  {
    id: generateId(),
    userId: '1',
    userEmail: 'patient@example.com',
    title: 'General Consultation Report',
    content: 'Patient presented with mild symptoms. Physical examination completed.',
    diagnosis: 'Common cold - viral infection',
    notes: 'Rest recommended. Stay hydrated. Follow up in 3 days if symptoms persist.',
    createdAt: new Date(Date.now() - 86400000).toISOString(),
    doctorName: 'Dr. Emily Johnson'
  },
  {
    id: generateId(),
    userId: '1',
    userEmail: 'patient@example.com',
    title: 'Annual Health Checkup',
    content: 'Complete blood work done. All vitals within normal range.',
    diagnosis: 'Healthy - no concerns',
    notes: 'Continue current lifestyle. Next checkup in 12 months.',
    createdAt: new Date(Date.now() - 2592000000).toISOString(),
    doctorName: 'Dr. Michael Chen'
  }
];

const createMockRooms = (): IoTRoom[] => [
  {
    id: generateId(),
    roomId: 'DOC-101',
    roomType: 'doctor',
    status: 'available',
    timestamp: new Date().toISOString(),
    location: 'First Floor - Wing A'
  },
  {
    id: generateId(),
    roomId: 'THR-201',
    roomType: 'therapy',
    status: 'busy',
    timestamp: new Date().toISOString(),
    location: 'Second Floor - Wing B'
  },
  {
    id: generateId(),
    roomId: 'EQP-301',
    roomType: 'equipment',
    status: 'available',
    timestamp: new Date().toISOString(),
    location: 'Third Floor - Radiology'
  },
  {
    id: generateId(),
    roomId: 'DOC-102',
    roomType: 'doctor',
    status: 'busy',
    timestamp: new Date().toISOString(),
    location: 'First Floor - Wing A'
  },
  {
    id: generateId(),
    roomId: 'THR-202',
    roomType: 'therapy',
    status: 'available',
    timestamp: new Date().toISOString(),
    location: 'Second Floor - Wing B'
  }
];

// Local storage keys
const STORAGE_KEYS = {
  requests: 'zetatech_requests',
  reports: 'zetatech_reports',
  rooms: 'zetatech_rooms'
};

export const useData = () => {
  const [requests, setRequests] = useState<ServiceRequest[]>([]);
  const [reports, setReports] = useState<MedicalReport[]>([]);
  const [rooms, setRooms] = useState<IoTRoom[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize data from localStorage or create mock data
  useEffect(() => {
    const storedRequests = localStorage.getItem(STORAGE_KEYS.requests);
    const storedReports = localStorage.getItem(STORAGE_KEYS.reports);
    const storedRooms = localStorage.getItem(STORAGE_KEYS.rooms);

    if (storedRequests) {
      setRequests(JSON.parse(storedRequests));
    } else {
      const mockRequests = createMockRequests();
      setRequests(mockRequests);
      localStorage.setItem(STORAGE_KEYS.requests, JSON.stringify(mockRequests));
    }

    if (storedReports) {
      setReports(JSON.parse(storedReports));
    } else {
      const mockReports = createMockReports();
      setReports(mockReports);
      localStorage.setItem(STORAGE_KEYS.reports, JSON.stringify(mockReports));
    }

    if (storedRooms) {
      setRooms(JSON.parse(storedRooms));
    } else {
      const mockRooms = createMockRooms();
      setRooms(mockRooms);
      localStorage.setItem(STORAGE_KEYS.rooms, JSON.stringify(mockRooms));
    }

    setIsInitialized(true);
  }, []);

  // Persist data changes
  useEffect(() => {
    if (isInitialized) {
      localStorage.setItem(STORAGE_KEYS.requests, JSON.stringify(requests));
    }
  }, [requests, isInitialized]);

  useEffect(() => {
    if (isInitialized) {
      localStorage.setItem(STORAGE_KEYS.reports, JSON.stringify(reports));
    }
  }, [reports, isInitialized]);

  useEffect(() => {
    if (isInitialized) {
      localStorage.setItem(STORAGE_KEYS.rooms, JSON.stringify(rooms));
    }
  }, [rooms, isInitialized]);

  // Request operations
  const createRequest = useCallback((userId: string, userEmail: string, serviceType: ServiceType): ServiceRequest => {
    const newRequest: ServiceRequest = {
      id: generateId(),
      userId,
      userEmail,
      serviceType,
      status: 'pending',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    setRequests(prev => [newRequest, ...prev]);
    return newRequest;
  }, []);

  const updateRequestStatus = useCallback((requestId: string, status: RequestStatus, reason?: string) => {
    setRequests(prev => prev.map(req => 
      req.id === requestId 
        ? { ...req, status, reason, updatedAt: new Date().toISOString() }
        : req
    ));
  }, []);

  const getUserRequests = useCallback((userId: string) => {
    return requests.filter(req => req.userId === userId);
  }, [requests]);

  const getAllRequests = useCallback(() => {
    return requests;
  }, [requests]);

  // Report operations
  const getUserReports = useCallback((userId: string) => {
    return reports.filter(rep => rep.userId === userId);
  }, [reports]);

  const createReport = useCallback((report: Omit<MedicalReport, 'id' | 'createdAt'>) => {
    const newReport: MedicalReport = {
      ...report,
      id: generateId(),
      createdAt: new Date().toISOString()
    };
    setReports(prev => [newReport, ...prev]);
    return newReport;
  }, []);

  // Room operations
  const getAllRooms = useCallback(() => {
    return rooms;
  }, [rooms]);

  const updateRoomStatus = useCallback((roomId: string, status: 'available' | 'busy') => {
    setRooms(prev => prev.map(room =>
      room.roomId === roomId
        ? { ...room, status, timestamp: new Date().toISOString() }
        : room
    ));
  }, []);

  // Stats
  const getDashboardStats = useCallback((): DashboardStats => {
    return {
      totalRequests: requests.length,
      pendingRequests: requests.filter(r => r.status === 'pending').length,
      approvedRequests: requests.filter(r => r.status === 'approved').length,
      rejectedRequests: requests.filter(r => r.status === 'rejected').length,
      totalUsers: new Set(requests.map(r => r.userId)).size + 2,
      activeRooms: rooms.filter(r => r.status === 'available').length
    };
  }, [requests, rooms]);

  return {
    requests,
    reports,
    rooms,
    isInitialized,
    createRequest,
    updateRequestStatus,
    getUserRequests,
    getAllRequests,
    getUserReports,
    createReport,
    getAllRooms,
    updateRoomStatus,
    getDashboardStats
  };
};
