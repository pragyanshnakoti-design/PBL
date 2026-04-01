import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  User, 
  Calendar, 
  FileText, 
  LogOut,
  Plus,
  Stethoscope,
  HeartPulse,
  CalendarDays,
  Eye,
  Shield
} from 'lucide-react';
import type { User as UserType, ServiceRequest, MedicalReport, ServiceType, RequestStatus } from '@/types';

interface UserDashboardProps {
  user: UserType;
  requests: ServiceRequest[];
  reports: MedicalReport[];
  onLogout: () => void;
  onCreateRequest: (serviceType: ServiceType) => void;
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

const getServiceIcon = (type: ServiceType) => {
  switch (type) {
    case 'consultation':
      return <Stethoscope className="w-5 h-5" />;
    case 'therapy':
      return <HeartPulse className="w-5 h-5" />;
    case 'followup':
      return <CalendarDays className="w-5 h-5" />;
  }
};

const getServiceLabel = (type: ServiceType) => {
  switch (type) {
    case 'consultation':
      return 'Doctor Consultation';
    case 'therapy':
      return 'Therapy Session';
    case 'followup':
      return 'Follow-up Appointment';
  }
};

export const UserDashboard = ({ user, requests, reports, onLogout, onCreateRequest }: UserDashboardProps) => {
  const [showNewRequestDialog, setShowNewRequestDialog] = useState(false);
  const [selectedService, setSelectedService] = useState<ServiceType>('consultation');
  const [selectedReport, setSelectedReport] = useState<MedicalReport | null>(null);

  const handleCreateRequest = () => {
    onCreateRequest(selectedService);
    setShowNewRequestDialog(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      {/* Header */}
      <header className="bg-white dark:bg-slate-900 border-b sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
              <User className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-slate-900 dark:text-slate-100">Patient Portal</h1>
              <p className="text-xs text-muted-foreground">{user.email}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="bg-cyan-50 dark:bg-cyan-950/30 text-cyan-700 dark:text-cyan-400">
              <Shield className="w-3 h-3 mr-1" />
              User
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
        <Card className="mb-8 bg-gradient-to-r from-cyan-500 to-blue-600 border-0 text-white">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold mb-1">Welcome, {user.name}</h2>
                <p className="text-cyan-100">Manage your healthcare services securely</p>
              </div>
              <div className="hidden sm:block">
                <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center">
                  <User className="w-8 h-8 text-white" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Tabs defaultValue="requests" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2 max-w-md">
            <TabsTrigger value="requests">
              <Calendar className="w-4 h-4 mr-2" />
              My Requests
            </TabsTrigger>
            <TabsTrigger value="reports">
              <FileText className="w-4 h-4 mr-2" />
              My Reports
            </TabsTrigger>
          </TabsList>

          {/* Requests Tab */}
          <TabsContent value="requests" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Service Requests</h3>
                <p className="text-sm text-muted-foreground">Track your service request status</p>
              </div>
              <Button onClick={() => setShowNewRequestDialog(true)}>
                <Plus className="w-4 h-4 mr-2" />
                New Request
              </Button>
            </div>

            {requests.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="p-12 text-center">
                  <Calendar className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 dark:text-slate-100 mb-1">No requests yet</h3>
                  <p className="text-sm text-muted-foreground mb-4">Create your first service request</p>
                  <Button onClick={() => setShowNewRequestDialog(true)}>
                    <Plus className="w-4 h-4 mr-2" />
                    Create Request
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4">
                {requests.map((request) => (
                  <Card key={request.id} className="hover:shadow-md transition-shadow">
                    <CardContent className="p-5">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4">
                          <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                            request.status === 'approved' ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600' :
                            request.status === 'rejected' ? 'bg-rose-100 dark:bg-rose-900/30 text-rose-600' :
                            'bg-amber-100 dark:bg-amber-900/30 text-amber-600'
                          }`}>
                            {getServiceIcon(request.serviceType)}
                          </div>
                          <div>
                            <h4 className="font-semibold text-slate-900 dark:text-slate-100">
                              {getServiceLabel(request.serviceType)}
                            </h4>
                            <p className="text-sm text-muted-foreground">
                              Requested on {new Date(request.createdAt).toLocaleDateString()}
                            </p>
                            {request.reason && (
                              <div className="mt-2 p-2 bg-slate-50 dark:bg-slate-800 rounded text-sm">
                                <span className="font-medium">Reason: </span>
                                {request.reason}
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex flex-col items-end gap-2">
                          {getStatusBadge(request.status)}
                          <span className="text-xs text-muted-foreground">
                            Updated: {new Date(request.updatedAt).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Reports Tab */}
          <TabsContent value="reports" className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold">Medical Reports</h3>
              <p className="text-sm text-muted-foreground">Your private medical reports - visible only to you</p>
            </div>

            {reports.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="p-12 text-center">
                  <FileText className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 dark:text-slate-100 mb-1">No reports yet</h3>
                  <p className="text-sm text-muted-foreground">Your medical reports will appear here</p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4">
                {reports.map((report) => (
                  <Card key={report.id} className="hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => setSelectedReport(report)}>
                    <CardContent className="p-5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-start gap-4">
                          <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 text-blue-600 rounded-lg flex items-center justify-center">
                            <FileText className="w-6 h-6" />
                          </div>
                          <div>
                            <h4 className="font-semibold text-slate-900 dark:text-slate-100">{report.title}</h4>
                            <p className="text-sm text-muted-foreground">
                              Dr. {report.doctorName} • {new Date(report.createdAt).toLocaleDateString()}
                            </p>
                            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1 line-clamp-1">
                              Diagnosis: {report.diagnosis}
                            </p>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm">
                          <Eye className="w-4 h-4 mr-2" />
                          View
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </main>

      {/* New Request Dialog */}
      <Dialog open={showNewRequestDialog} onOpenChange={setShowNewRequestDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Service Request</DialogTitle>
            <DialogDescription>
              Select the type of service you need
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">Service Type</label>
            <Select value={selectedService} onValueChange={(v) => setSelectedService(v as ServiceType)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="consultation">
                  <div className="flex items-center gap-2">
                    <Stethoscope className="w-4 h-4" />
                    Doctor Consultation
                  </div>
                </SelectItem>
                <SelectItem value="therapy">
                  <div className="flex items-center gap-2">
                    <HeartPulse className="w-4 h-4" />
                    Therapy Session
                  </div>
                </SelectItem>
                <SelectItem value="followup">
                  <div className="flex items-center gap-2">
                    <CalendarDays className="w-4 h-4" />
                    Follow-up Appointment
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewRequestDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateRequest}>
              Submit Request
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Report Detail Dialog */}
      <Dialog open={!!selectedReport} onOpenChange={() => setSelectedReport(null)}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>{selectedReport?.title}</DialogTitle>
            <DialogDescription>
              Medical Report • {selectedReport && new Date(selectedReport.createdAt).toLocaleDateString()}
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="max-h-[60vh]">
            {selectedReport && (
              <div className="space-y-6 py-4">
                <div className="flex items-center gap-2 p-3 bg-emerald-50 dark:bg-emerald-950/30 rounded-lg">
                  <Shield className="w-5 h-5 text-emerald-600" />
                  <span className="text-sm text-emerald-800 dark:text-emerald-400">
                    This report is private and visible only to you
                  </span>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Doctor</h4>
                  <p className="font-medium">{selectedReport.doctorName}</p>
                </div>

                <Separator />

                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Diagnosis</h4>
                  <p className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">{selectedReport.diagnosis}</p>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Examination Details</h4>
                  <p className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">{selectedReport.content}</p>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-muted-foreground mb-1">Doctor's Notes</h4>
                  <p className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">{selectedReport.notes}</p>
                </div>
              </div>
            )}
          </ScrollArea>
          <DialogFooter>
            <Button onClick={() => setSelectedReport(null)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
