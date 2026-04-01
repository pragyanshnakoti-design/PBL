import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { 
  Shield, 
  CheckCircle, 
  XCircle, 
  Clock,
  LogOut,
  Activity,
  Wifi,
  AlertTriangle,
  Eye,
  Building2
} from 'lucide-react';
import type { User as UserType, ServiceRequest, IoTRoom, RequestStatus, DashboardStats } from '@/types';

interface AdminDashboardProps {
  user: UserType;
  requests: ServiceRequest[];
  rooms: IoTRoom[];
  stats: DashboardStats;
  onLogout: () => void;
  onUpdateRequestStatus: (requestId: string, status: RequestStatus, reason?: string) => void;
}

const getStatusBadge = (status: RequestStatus) => {
  switch (status) {
    case 'approved':
      return <Badge className="bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400">Approved</Badge>;
    case 'rejected':
      return <Badge className="bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-400">Rejected</Badge>;
    default:
      return <Badge className="bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400">Pending</Badge>;
  }
};

const getServiceLabel = (type: string) => {
  switch (type) {
    case 'consultation':
      return 'Doctor Consultation';
    case 'therapy':
      return 'Therapy Session';
    case 'followup':
      return 'Follow-up Appointment';
    default:
      return type;
  }
};

const getRoomTypeLabel = (type: string) => {
  switch (type) {
    case 'doctor':
      return 'Doctor Room';
    case 'therapy':
      return 'Therapy Room';
    case 'equipment':
      return 'Equipment Room';
    default:
      return type;
  }
};

export const AdminDashboard = ({ user, requests, rooms, stats, onLogout, onUpdateRequestStatus }: AdminDashboardProps) => {
  const [selectedRequest, setSelectedRequest] = useState<ServiceRequest | null>(null);
  const [actionType, setActionType] = useState<'approve' | 'reject' | null>(null);
  const [reason, setReason] = useState('');

  const pendingRequests = requests.filter(r => r.status === 'pending');
  const processedRequests = requests.filter(r => r.status !== 'pending');

  const handleAction = () => {
    if (selectedRequest && actionType) {
      const status: RequestStatus = actionType === 'approve' ? 'approved' : 'rejected';
      onUpdateRequestStatus(selectedRequest.id, status, reason);
      setSelectedRequest(null);
      setActionType(null);
      setReason('');
    }
  };

  const openActionDialog = (request: ServiceRequest, action: 'approve' | 'reject') => {
    setSelectedRequest(request);
    setActionType(action);
    setReason('');
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      {/* Header */}
      <header className="bg-white dark:bg-slate-900 border-b sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-slate-900 dark:text-slate-100">Admin Dashboard</h1>
              <p className="text-xs text-muted-foreground">{user.email}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="bg-purple-50 dark:bg-purple-950/30 text-purple-700 dark:text-purple-400">
              <Shield className="w-3 h-3 mr-1" />
              Administrator
            </Badge>
            <Button variant="ghost" size="sm" onClick={onLogout}>
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Banner */}
        <Card className="mb-8 bg-gradient-to-r from-purple-500 to-indigo-600 border-0 text-white">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold mb-1">Admin Control Center</h2>
                <p className="text-purple-100">Manage service requests and monitor system status</p>
              </div>
              <div className="hidden sm:block">
                <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center">
                  <Building2 className="w-8 h-8 text-white" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Total Requests</p>
                  <p className="text-2xl font-bold">{stats?.totalRequests || 0}</p>
                </div>
                <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                  <Activity className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Pending</p>
                  <p className="text-2xl font-bold">{stats?.pendingRequests || 0}</p>
                </div>
                <div className="w-12 h-12 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center">
                  <Clock className="w-6 h-6 text-amber-600" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Approved</p>
                  <p className="text-2xl font-bold">{stats?.approvedRequests || 0}</p>
                </div>
                <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg flex items-center justify-center">
                  <CheckCircle className="w-6 h-6 text-emerald-600" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Active Rooms</p>
                  <p className="text-2xl font-bold">{stats?.activeRooms || 0}</p>
                </div>
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                  <Wifi className="w-6 h-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="pending" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 max-w-lg">
            <TabsTrigger value="pending">
              <Clock className="w-4 h-4 mr-2" />
              Pending ({pendingRequests.length})
            </TabsTrigger>
            <TabsTrigger value="processed">
              <CheckCircle className="w-4 h-4 mr-2" />
              Processed
            </TabsTrigger>
            <TabsTrigger value="iot">
              <Wifi className="w-4 h-4 mr-2" />
              IoT Rooms
            </TabsTrigger>
          </TabsList>

          {/* Pending Requests Tab */}
          <TabsContent value="pending" className="space-y-6">
            <div className="flex items-center gap-2 p-4 bg-amber-50 dark:bg-amber-950/30 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
              <span className="text-sm text-amber-800 dark:text-amber-400">
                <strong>Privacy Note:</strong> You can only see user email and request status. Medical reports are private.
              </span>
            </div>

            {pendingRequests.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="p-12 text-center">
                  <CheckCircle className="w-12 h-12 text-emerald-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 dark:text-slate-100 mb-1">All caught up!</h3>
                  <p className="text-sm text-muted-foreground">No pending requests to review</p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4">
                {pendingRequests.map((request) => (
                  <Card key={request.id}>
                    <CardContent className="p-5">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4">
                          <div className="w-12 h-12 bg-amber-100 dark:bg-amber-900/30 text-amber-600 rounded-lg flex items-center justify-center">
                            <Clock className="w-6 h-6" />
                          </div>
                          <div>
                            <h4 className="font-semibold text-slate-900 dark:text-slate-100">
                              {getServiceLabel(request.serviceType)}
                            </h4>
                            <p className="text-sm text-muted-foreground">
                              From: {request.userEmail}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              Requested: {new Date(request.createdAt).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => openActionDialog(request, 'reject')}
                            className="text-rose-600 border-rose-200 hover:bg-rose-50"
                          >
                            <XCircle className="w-4 h-4 mr-1" />
                            Reject
                          </Button>
                          <Button 
                            size="sm"
                            onClick={() => openActionDialog(request, 'approve')}
                            className="bg-emerald-600 hover:bg-emerald-700"
                          >
                            <CheckCircle className="w-4 h-4 mr-1" />
                            Approve
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Processed Requests Tab */}
          <TabsContent value="processed" className="space-y-6">
            <div className="flex items-center gap-2 p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
              <Eye className="w-5 h-5 text-blue-600" />
              <span className="text-sm text-blue-800 dark:text-blue-400">
                View all processed requests with decision reasons for transparency.
              </span>
            </div>

            {processedRequests.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="p-12 text-center">
                  <Activity className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 dark:text-slate-100 mb-1">No processed requests</h3>
                  <p className="text-sm text-muted-foreground">Approved and rejected requests will appear here</p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4">
                {processedRequests.map((request) => (
                  <Card key={request.id}>
                    <CardContent className="p-5">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4">
                          <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                            request.status === 'approved' 
                              ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600' 
                              : 'bg-rose-100 dark:bg-rose-900/30 text-rose-600'
                          }`}>
                            {request.status === 'approved' ? <CheckCircle className="w-6 h-6" /> : <XCircle className="w-6 h-6" />}
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <h4 className="font-semibold text-slate-900 dark:text-slate-100">
                                {getServiceLabel(request.serviceType)}
                              </h4>
                              {getStatusBadge(request.status)}
                            </div>
                            <p className="text-sm text-muted-foreground">
                              From: {request.userEmail}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              Processed: {new Date(request.updatedAt).toLocaleString()}
                            </p>
                            {request.reason && (
                              <div className={`mt-2 p-2 rounded text-sm ${
                                request.status === 'approved'
                                  ? 'bg-emerald-50 dark:bg-emerald-950/30 text-emerald-800 dark:text-emerald-400'
                                  : 'bg-rose-50 dark:bg-rose-950/30 text-rose-800 dark:text-rose-400'
                              }`}>
                                <span className="font-medium">Reason: </span>
                                {request.reason}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* IoT Rooms Tab */}
          <TabsContent value="iot" className="space-y-6">
            <div className="flex items-center gap-2 p-4 bg-purple-50 dark:bg-purple-950/30 rounded-lg">
              <Wifi className="w-5 h-5 text-purple-600" />
              <span className="text-sm text-purple-800 dark:text-purple-400">
                <strong>IoT Monitoring:</strong> Real-time room availability status from connected devices.
              </span>
            </div>

            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {rooms.map((room) => (
                <Card key={room.id}>
                  <CardContent className="p-5">
                    <div className="flex items-start justify-between mb-4">
                      <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 text-purple-600 rounded-lg flex items-center justify-center">
                        <Building2 className="w-5 h-5" />
                      </div>
                      <Badge className={room.status === 'available' 
                        ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400' 
                        : 'bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-400'
                      }>
                        {room.status === 'available' ? 'Available' : 'Busy'}
                      </Badge>
                    </div>
                    <h4 className="font-semibold text-slate-900 dark:text-slate-100">{room.roomId}</h4>
                    <p className="text-sm text-muted-foreground">{getRoomTypeLabel(room.roomType)}</p>
                    <p className="text-xs text-muted-foreground mt-1">{room.location}</p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Last update: {new Date(room.timestamp).toLocaleTimeString()}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Action Dialog */}
      <Dialog open={!!selectedRequest && !!actionType} onOpenChange={() => { setSelectedRequest(null); setActionType(null); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {actionType === 'approve' ? 'Approve Request' : 'Reject Request'}
            </DialogTitle>
            <DialogDescription>
              {selectedRequest && `Add a reason for ${actionType}ing the ${getServiceLabel(selectedRequest.serviceType)} request from ${selectedRequest.userEmail}`}
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">
              Reason <span className="text-muted-foreground">(optional)</span>
            </label>
            <Textarea
              placeholder={actionType === 'approve' ? 'e.g., Doctor available on requested date' : 'e.g., Please reschedule for next week'}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={3}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setSelectedRequest(null); setActionType(null); }}>
              Cancel
            </Button>
            <Button 
              onClick={handleAction}
              className={actionType === 'approve' ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-rose-600 hover:bg-rose-700'}
            >
              {actionType === 'approve' ? 'Approve' : 'Reject'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
