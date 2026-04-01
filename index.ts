export type UserRole = 'user' | 'admin';

export type RequestStatus = 'pending' | 'approved' | 'rejected';

export type ServiceType = 'consultation' | 'therapy' | 'followup';

export type RoomStatus = 'available' | 'busy';

export type RoomType = 'therapy' | 'doctor' | 'equipment';

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  passwordHash: string;
}

export interface ServiceRequest {
  id: string;
  userId: string;
  userEmail: string;
  serviceType: ServiceType;
  status: RequestStatus;
  reason?: string;
  createdAt: string;
  updatedAt: string;
}

export interface MedicalReport {
  id: string;
  userId: string;
  userEmail: string;
  title: string;
  content: string;
  diagnosis: string;
  notes: string;
  createdAt: string;
  doctorName: string;
}

export interface IoTRoom {
  id: string;
  roomId: string;
  roomType: RoomType;
  status: RoomStatus;
  timestamp: string;
  location: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
}

export interface DashboardStats {
  totalRequests: number;
  pendingRequests: number;
  approvedRequests: number;
  rejectedRequests: number;
  totalUsers: number;
  activeRooms: number;
}
