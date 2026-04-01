import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Shield, 
  Cloud, 
  Activity, 
  Lock, 
  UserCheck, 
  Stethoscope, 
  FileText,
  Wifi,
  CheckCircle,
  AlertCircle,
  Eye,
  EyeOff,
  Building2,
  HeartPulse,
  Cpu
} from 'lucide-react';

interface LandingPageProps {
  onLogin: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
}

export const LandingPage = ({ onLogin }: LandingPageProps) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'login' | 'features' | 'security'>('login');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    
    const result = await onLogin(email, password);
    
    if (!result.success) {
      setError(result.error || 'Login failed');
    }
    
    setIsLoading(false);
  };

  const quickLogin = (type: 'user' | 'admin') => {
    if (type === 'user') {
      setEmail('patient@example.com');
      setPassword('password123');
    } else {
      setEmail('admin@zetatech.com');
      setPassword('admin456');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Header */}
      <header className="border-b bg-white/80 dark:bg-slate-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
              <Building2 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
                ZetaTech
              </h1>
              <p className="text-xs text-muted-foreground">Hospital Service Management</p>
            </div>
          </div>
          <Badge variant="outline" className="bg-cyan-50 dark:bg-cyan-950/30 text-cyan-700 dark:text-cyan-400 border-cyan-200 dark:border-cyan-800">
            <Cloud className="w-3 h-3 mr-1" />
            Cloud-Based
          </Badge>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          {/* Left Column - Info */}
          <div className="space-y-8">
            <div className="space-y-4">
              <Badge className="bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 hover:bg-blue-100">
                Academic Prototype
              </Badge>
              <h2 className="text-4xl font-bold text-slate-900 dark:text-slate-100 leading-tight">
                Cloud-Based Secure
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-500 to-blue-600">
                  Hospital Management
                </span>
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-400">
                A secure, cloud-based platform for managing hospital services with 
                role-based access control and IoT integration.
              </p>
            </div>

            {/* Navigation Tabs */}
            <div className="flex gap-2 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg w-fit">
              <button
                onClick={() => setActiveTab('login')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'login' 
                    ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-slate-100' 
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Login
              </button>
              <button
                onClick={() => setActiveTab('features')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'features' 
                    ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-slate-100' 
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Features
              </button>
              <button
                onClick={() => setActiveTab('security')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'security' 
                    ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-slate-100' 
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Security
              </button>
            </div>

            {/* Tab Content */}
            {activeTab === 'login' && (
              <div className="space-y-4 animate-fadeIn">
                <Card className="border-slate-200 dark:border-slate-800">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <UserCheck className="w-5 h-5 text-cyan-500" />
                      Quick Login
                    </CardTitle>
                    <CardDescription>
                      Click to auto-fill demo credentials
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex gap-3">
                    <Button 
                      variant="outline" 
                      onClick={() => quickLogin('user')}
                      className="flex-1"
                    >
                      <UserCheck className="w-4 h-4 mr-2" />
                      Demo User
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={() => quickLogin('admin')}
                      className="flex-1"
                    >
                      <Shield className="w-4 h-4 mr-2" />
                      Demo Admin
                    </Button>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'features' && (
              <div className="grid sm:grid-cols-2 gap-4 animate-fadeIn">
                <Card className="border-slate-200 dark:border-slate-800">
                  <CardContent className="pt-6">
                    <Stethoscope className="w-8 h-8 text-cyan-500 mb-3" />
                    <h3 className="font-semibold mb-1">Service Requests</h3>
                    <p className="text-sm text-muted-foreground">
                      Request consultations, therapy sessions, and follow-ups
                    </p>
                  </CardContent>
                </Card>
                <Card className="border-slate-200 dark:border-slate-800">
                  <CardContent className="pt-6">
                    <FileText className="w-8 h-8 text-emerald-500 mb-3" />
                    <h3 className="font-semibold mb-1">Medical Reports</h3>
                    <p className="text-sm text-muted-foreground">
                      View your personal medical reports securely
                    </p>
                  </CardContent>
                </Card>
                <Card className="border-slate-200 dark:border-slate-800">
                  <CardContent className="pt-6">
                    <Wifi className="w-8 h-8 text-purple-500 mb-3" />
                    <h3 className="font-semibold mb-1">IoT Monitoring</h3>
                    <p className="text-sm text-muted-foreground">
                      Real-time room availability status
                    </p>
                  </CardContent>
                </Card>
                <Card className="border-slate-200 dark:border-slate-800">
                  <CardContent className="pt-6">
                    <Activity className="w-8 h-8 text-rose-500 mb-3" />
                    <h3 className="font-semibold mb-1">Admin Dashboard</h3>
                    <p className="text-sm text-muted-foreground">
                      Manage requests and generate reports
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}

            {activeTab === 'security' && (
              <div className="space-y-4 animate-fadeIn">
                <Card className="border-slate-200 dark:border-slate-800">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Lock className="w-5 h-5 text-emerald-500" />
                      Security Features
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-emerald-500 mt-0.5" />
                      <div>
                        <p className="font-medium">Password Hashing</p>
                        <p className="text-sm text-muted-foreground">Secure password storage with hashing</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-emerald-500 mt-0.5" />
                      <div>
                        <p className="font-medium">JWT Authentication</p>
                        <p className="text-sm text-muted-foreground">Token-based secure session management</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-emerald-500 mt-0.5" />
                      <div>
                        <p className="font-medium">Role-Based Access</p>
                        <p className="text-sm text-muted-foreground">Different permissions for users and admins</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-slate-200 dark:border-slate-800 bg-amber-50 dark:bg-amber-950/20">
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5" />
                      <div>
                        <p className="font-medium text-amber-800 dark:text-amber-300">Privacy Notice</p>
                        <p className="text-sm text-amber-700 dark:text-amber-400">
                          User medical data is private and visible only to the user. 
                          Admin access is limited to maintain data confidentiality.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* Right Column - Login Form */}
          <div className="lg:sticky lg:top-24">
            <Card className="border-slate-200 dark:border-slate-800 shadow-xl">
              <CardHeader className="space-y-1">
                <CardTitle className="text-2xl text-center">Welcome Back</CardTitle>
                <CardDescription className="text-center">
                  Sign in to access your dashboard
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Email</label>
                    <Input
                      type="email"
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      className="h-11"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Password</label>
                    <div className="relative">
                      <Input
                        type={showPassword ? 'text' : 'password'}
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        className="h-11 pr-10"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>

                  {error && (
                    <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                      <AlertCircle className="w-4 h-4" />
                      {error}
                    </div>
                  )}

                  <Button 
                    type="submit" 
                    className="w-full h-11 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
                    disabled={isLoading}
                  >
                    {isLoading ? 'Signing in...' : 'Sign In'}
                  </Button>
                </form>

                <Separator className="my-6" />

                <div className="text-center text-sm text-muted-foreground">
                  <p className="mb-2">Demo Credentials:</p>
                  <div className="space-y-1 text-xs">
                    <p><strong>User:</strong> patient@example.com / password123</p>
                    <p><strong>Admin:</strong> admin@zetatech.com / admin456</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* System Status */}
            <div className="mt-6 grid grid-cols-3 gap-3">
              <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
                <HeartPulse className="w-5 h-5 text-rose-500 mx-auto mb-1" />
                <p className="text-xs text-muted-foreground">Services</p>
                <p className="font-semibold text-sm">Active</p>
              </div>
              <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
                <Cloud className="w-5 h-5 text-cyan-500 mx-auto mb-1" />
                <p className="text-xs text-muted-foreground">Cloud</p>
                <p className="font-semibold text-sm">Online</p>
              </div>
              <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
                <Cpu className="w-5 h-5 text-purple-500 mx-auto mb-1" />
                <p className="text-xs text-muted-foreground">IoT</p>
                <p className="font-semibold text-sm">Connected</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-16 py-8 bg-white dark:bg-slate-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-muted-foreground">
          <p>© 2026 ZetaTech. Academic Prototype - Hospital Service Management System.</p>
          <p className="mt-1">This is an academic prototype focused on system design and security.</p>
        </div>
      </footer>
    </div>
  );
};
