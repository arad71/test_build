import React, { useState, useEffect } from 'react';
import { Calendar, FileText, Clock, CheckCircle, AlertTriangle, Users, Building, Search, Plus, Eye, Edit, Trash2, Download } from 'lucide-react';

const BuildingApprovalSystem = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [applications, setApplications] = useState([]);
  const [selectedApplication, setSelectedApplication] = useState(null);
  const [showNewApplicationModal, setShowNewApplicationModal] = useState(false);
  const [showFormCheckerModal, setShowFormCheckerModal] = useState(false);
  const [showSystemReviewModal, setShowSystemReviewModal] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showNotificationsPanel, setShowNotificationsPanel] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [currentUser, setCurrentUser] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [auditLog, setAuditLog] = useState([]);

  // Production-ready user roles and permissions
  const userRoles = {
    'admin': {
      name: 'System Administrator',
      permissions: ['all'],
      color: 'bg-purple-100 text-purple-800'
    },
    'senior_officer': {
      name: 'Senior Planning Officer',
      permissions: ['approve', 'reject', 'assign', 'review', 'create', 'audit'],
      color: 'bg-blue-100 text-blue-800'
    },
    'planning_officer': {
      name: 'Planning Officer',
      permissions: ['review', 'recommend', 'create', 'update'],
      color: 'bg-green-100 text-green-800'
    },
    'clerical': {
      name: 'Clerical Officer',
      permissions: ['create', 'update', 'view'],
      color: 'bg-gray-100 text-gray-800'
    },
    'applicant': {
      name: 'External Applicant',
      permissions: ['create', 'view_own', 'submit'],
      color: 'bg-yellow-100 text-yellow-800'
    }
  };

  // Mock user database
  const mockUsers = [
    {
      id: 1,
      username: 'sarah.johnson',
      fullName: 'Sarah Johnson',
      email: 'sarah.johnson@kalamunda.wa.gov.au',
      role: 'senior_officer',
      department: 'Development Services',
      lastLogin: '2025-06-21T08:30:00Z',
      isActive: true
    },
    {
      id: 2,
      username: 'mike.chen',
      fullName: 'Mike Chen',
      email: 'mike.chen@kalamunda.wa.gov.au',
      role: 'planning_officer',
      department: 'Planning',
      lastLogin: '2025-06-21T09:15:00Z',
      isActive: true
    },
    {
      id: 3,
      username: 'admin',
      fullName: 'System Administrator',
      email: 'admin@kalamunda.wa.gov.au',
      role: 'admin',
      department: 'IT',
      lastLogin: '2025-06-21T07:00:00Z',
      isActive: true
    }
  ];

  // Initialize with demo user for testing
  useEffect(() => {
    // Auto-login for demo purposes
    setCurrentUser(mockUsers[0]);
    
    // Initialize sample applications
    const sampleApplications = [
      {
        id: 'DA2025001',
        type: 'Development Approval',
        property: '123 Lesmurdie Road, Lesmurdie',
        applicant: 'John Smith',
        description: 'Two-storey residential extension',
        status: 'Officer Assessment',
        submissionDate: '2025-05-15',
        targetDate: '2025-07-15',
        assignedOfficer: 'Sarah Johnson',
        referrals: ['Engineering', 'Environmental Health'],
        publicNotification: true,
        documents: ['Site Plan', 'Floor Plans', 'Elevations', 'Landscape Plan', 'Traffic Report', 'Drainage Plan']
      },
      {
        id: 'BP2025002',
        type: 'Building Permit - Certified (BA01)',
        property: '45 Kalamunda Road, Kalamunda',
        applicant: 'ABC Construction',
        description: 'New commercial building',
        status: 'External Referral',
        submissionDate: '2025-06-01',
        targetDate: '2025-08-30',
        assignedOfficer: 'Mike Chen',
        referrals: ['Main Roads WA', 'Dept of Fire & Emergency'],
        publicNotification: false,
        documents: ['Certified Plans', 'Site Plan', 'Floor Plans', 'Elevations', 'Structural Report', 'Fire Safety Plan', 'Certificate of Design Compliance (BA03)', 'Traffic Report']
      },
      {
        id: 'DP2025003',
        type: 'Demolition Permit (BA05)',
        property: '78 Welshpool Road, Welshpool',
        applicant: 'Demo Pro Pty Ltd',
        description: 'Demolition of existing warehouse',
        status: 'Approved',
        submissionDate: '2025-04-20',
        targetDate: '2025-06-20',
        assignedOfficer: 'Lisa Wong',
        referrals: ['Environmental Health'],
        publicNotification: false,
        documents: ['Demolition Plan', 'Asbestos Report', 'Traffic Management'],
        systemReview: {
          overallScore: 89,
          documentScore: 100,
          fieldCompleteness: 100,
          riskLevel: 2,
          systemRecommendations: ['Application appears ready for approval - recommend expedited review'],
          reviewDate: '2025-04-21'
        }
      },
      {
        id: 'BA2025004',
        type: 'Building Permit - Uncertified (BA02)',
        property: '22 Forest Road, Forrestfield',
        applicant: 'Green Homes Pty Ltd',
        description: 'Single storey residential dwelling',
        status: 'DCU Review',
        submissionDate: '2025-06-10',
        targetDate: '2025-08-10',
        assignedOfficer: 'David Lee',
        referrals: ['Planning', 'Engineering'],
        publicNotification: true,
        documents: ['Architectural Plans', 'Site Plan', 'Floor Plans', 'Site Analysis', 'Energy Report']
      }
    ];
    setApplications(sampleApplications);
    
    // Initialize sample notifications
    setNotifications([
      {
        id: 1,
        type: 'warning',
        title: 'Application Overdue',
        message: 'Application DA2025001 is 2 days overdue for review',
        applicationId: 'DA2025001',
        timestamp: new Date().toISOString(),
        read: false
      },
      {
        id: 2,
        type: 'success',
        title: 'System Backup Complete',
        message: 'Daily system backup completed successfully',
        timestamp: new Date().toISOString(),
        read: false
      }
    ]);

    // Initialize audit log
    setAuditLog([
      {
        id: 1,
        timestamp: new Date().toISOString(),
        user: 'Sarah Johnson',
        action: 'application_approved',
        description: 'Approved application DP2025003',
        applicationId: 'DP2025003',
        ipAddress: '192.168.1.100'
      }
    ]);
  }, []);

  // Authentication and security functions
  const hasPermission = (permission) => {
    if (!currentUser) return false;
    const userRole = userRoles[currentUser.role];
    return userRole.permissions.includes('all') || userRole.permissions.includes(permission);
  };

  const login = (username, password) => {
    // Simulate authentication
    const user = mockUsers.find(u => u.username === username);
    if (user && password === 'demo123') {
      setCurrentUser(user);
      addAuditEntry('user_login', `User ${user.fullName} logged in`);
      setShowLoginModal(false);
      return true;
    }
    return false;
  };

  const logout = () => {
    if (currentUser) {
      addAuditEntry('user_logout', `User ${currentUser.fullName} logged out`);
      setCurrentUser(null);
      setActiveTab('dashboard');
    }
  };

  // Audit logging system
  const addAuditEntry = (action, description, applicationId = null) => {
    const entry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      user: currentUser?.fullName || 'System',
      action,
      description,
      applicationId,
      ipAddress: '192.168.1.100', // Simulated
      userAgent: navigator.userAgent.split(' ')[0] // Simplified
    };
    setAuditLog(prev => [entry, ...prev.slice(0, 99)]); // Keep last 100 entries
  };

  // Notification system
  const addNotification = (type, title, message, applicationId = null) => {
    const notification = {
      id: Date.now(),
      type, // 'info', 'success', 'warning', 'error'
      title,
      message,
      applicationId,
      timestamp: new Date().toISOString(),
      read: false
    };
    setNotifications(prev => [notification, ...prev]);

    // Auto-remove after 5 seconds for non-error notifications
    if (type !== 'error') {
      setTimeout(() => {
        setNotifications(prev => prev.filter(n => n.id !== notification.id));
      }, 5000);
    }
  };

  // Email notification simulation
  const sendEmailNotification = (to, subject, body, applicationId = null) => {
    console.log('📧 Email Notification:', { to, subject, body, applicationId });
    addNotification('info', 'Email Sent', `Notification sent to ${to}`, applicationId);
    addAuditEntry('email_sent', `Email notification sent to ${to} regarding ${subject}`, applicationId);
  };

  // PDF generation simulation
  const generatePDF = (type, application) => {
    addNotification('info', 'PDF Generated', `${type} report created for ${application.id}`);
    addAuditEntry('pdf_generated', `${type} PDF generated for application ${application.id}`, application.id);
    
    // Simulate PDF download
    const blob = new Blob([`${type} Report for ${application.id}\n\nApplication Details:\nType: ${application.type}\nProperty: ${application.property}\nApplicant: ${application.applicant}\nStatus: ${application.status}\n\nGenerated: ${new Date().toLocaleString()}`], 
      { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${application.id}-${type.toLowerCase()}-report.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Performance monitoring
  const performanceMetrics = {
    averageProcessingTime: '4.2 days',
    approvalRate: '87%',
    systemUptime: '99.9%',
    userSatisfaction: '4.6/5',
    totalApplications: applications.length,
    pendingApplications: applications.filter(app => !['Approved', 'Refused'].includes(app.status)).length
  };

  const applicationStatuses = [
    'Draft', 'Submitted', 'DCU Review', 'Internal Referral', 'External Referral',
    'Public Notification', 'Officer Assessment', 'Awaiting Council', 'Approved', 'Refused', 'Appealed'
  ];

  const applicationTypes = [
    'Development Approval',
    'Building Permit - Certified (BA01)',
    'Building Permit - Uncertified (BA02)',
    'Demolition Permit (BA05)',
    'Occupancy Permit (BA09)',
    'Building Approval Certificate (BA13)',
    'Amendment - Building Permit/Builder Details (BA19)',
    'Extension - Building/Demolition Permit (BA22)',
    'Certificate of Design Compliance (BA03)',
    'Certificate of Construction Compliance (BA17)',
    'Notice of Completion (BA07)'
  ];

  const requiredDocuments = {
    'Development Approval': [
      'Site Plan', 'Floor Plans', 'Elevations', 'Landscape Plan', 
      'Statement of Compliance', 'Traffic Impact Assessment', 'Drainage Plan'
    ],
    'Building Permit - Certified (BA01)': [
      'Certified Plans', 'Structural Calculations', 'Energy Efficiency Report', 
      'Plumbing Plans', 'Electrical Plans', 'Certificate of Design Compliance (BA03)'
    ],
    'Building Permit - Uncertified (BA02)': [
      'Architectural Plans', 'Structural Plans', 'Building Services Plans', 
      'Site Analysis', 'Specification Schedule', 'Energy Efficiency Report'
    ],
    'Demolition Permit (BA05)': [
      'Demolition Plan', 'Asbestos Survey Report', 'Traffic Management Plan', 
      'Waste Management Plan', 'Site Safety Plan'
    ],
    'Occupancy Permit (BA09)': [
      'Certificate of Construction Compliance (BA17)', 'Fire Safety Certificate', 
      'Disability Access Certificate', 'Final Inspection Report'
    ],
    'Building Approval Certificate (BA13)': [
      'Certificate of Building Compliance (BA18)', 'As-Built Plans', 
      'Structural Assessment', 'Compliance Statement'
    ]
  };

  const getStatusColor = (status) => {
    const colors = {
      'Draft': 'bg-gray-100 text-gray-800',
      'Submitted': 'bg-blue-100 text-blue-800',
      'DCU Review': 'bg-purple-100 text-purple-800',
      'Internal Referral': 'bg-yellow-100 text-yellow-800',
      'External Referral': 'bg-orange-100 text-orange-800',
      'Public Notification': 'bg-cyan-100 text-cyan-800',
      'Officer Assessment': 'bg-indigo-100 text-indigo-800',
      'Awaiting Council': 'bg-pink-100 text-pink-800',
      'Approved': 'bg-green-100 text-green-800',
      'Refused': 'bg-red-100 text-red-800',
      'Appealed': 'bg-red-200 text-red-900'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  const getDaysRemaining = (targetDate) => {
    const today = new Date();
    const target = new Date(targetDate);
    const diffTime = target - today;
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  const filteredApplications = applications.filter(app => {
    const matchesSearch = app.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         app.property.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         app.applicant.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === 'all' || app.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  const LoginModal = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleLogin = () => {
      if (login(username, password)) {
        setUsername('');
        setPassword('');
        setError('');
      } else {
        setError('Invalid credentials. Use demo credentials: any username with password "demo123"');
      }
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-lg max-w-md w-full p-6">
          <div className="text-center mb-6">
            <Building className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900">City of Kalamunda</h2>
            <p className="text-gray-600">Building Approval System Login</p>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Username</label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
              <input
                type="password"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
              />
            </div>
            
            {error && (
              <div className="bg-red-50 border border-red-200 rounded p-3 text-red-700 text-sm">
                {error}
              </div>
            )}
            
            <div className="bg-blue-50 border border-blue-200 rounded p-3 text-blue-700 text-sm">
              <p className="font-medium">Demo Credentials:</p>
              <p>Username: sarah.johnson (or any user)</p>
              <p>Password: demo123</p>
            </div>
            
            <button
              onClick={handleLogin}
              className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700"
            >
              Login
            </button>
          </div>
        </div>
      </div>
    );
  };

  const NotificationsPanel = () => (
    <div className="fixed top-16 right-4 w-80 bg-white rounded-lg shadow-lg border z-40 max-h-96 overflow-y-auto">
      <div className="p-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h3 className="font-semibold text-gray-900">Notifications</h3>
          <button
            onClick={() => setShowNotificationsPanel(false)}
            className="text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
        </div>
      </div>
      
      <div className="p-4 space-y-3">
        {notifications.length === 0 ? (
          <p className="text-gray-500 text-center py-4">No new notifications</p>
        ) : (
          notifications.slice(0, 10).map(notification => (
            <div key={notification.id} className={`p-3 rounded-lg border-l-4 ${
              notification.type === 'error' ? 'border-red-500 bg-red-50' :
              notification.type === 'warning' ? 'border-yellow-500 bg-yellow-50' :
              notification.type === 'success' ? 'border-green-500 bg-green-50' :
              'border-blue-500 bg-blue-50'
            }`}>
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <p className="font-medium text-gray-900 text-sm">{notification.title}</p>
                  <p className="text-gray-600 text-xs mt-1">{notification.message}</p>
                  <p className="text-gray-400 text-xs mt-1">
                    {new Date(notification.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const DashboardView = () => {
    const statusCounts = applicationStatuses.reduce((acc, status) => {
      acc[status] = applications.filter(app => app.status === status).length;
      return acc;
    }, {});

    const overdueApplications = applications.filter(app => getDaysRemaining(app.targetDate) < 0);
    const urgentApplications = applications.filter(app => {
      const days = getDaysRemaining(app.targetDate);
      return days >= 0 && days <= 7;
    });

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center">
              <FileText className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Applications</p>
                <p className="text-2xl font-bold text-gray-900">{applications.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center">
              <Clock className="h-8 w-8 text-orange-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Urgent (≤7 days)</p>
                <p className="text-2xl font-bold text-orange-600">{urgentApplications.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center">
              <AlertTriangle className="h-8 w-8 text-red-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Overdue</p>
                <p className="text-2xl font-bold text-red-600">{overdueApplications.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center">
              <CheckCircle className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Approved This Month</p>
                <p className="text-2xl font-bold text-green-600">{statusCounts.Approved || 0}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Applications by Status</h3>
            <div className="space-y-3">
              {applicationStatuses.map(status => (
                <div key={status} className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">{status}</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(status)}`}>
                    {statusCounts[status] || 0}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
            <div className="space-y-4">
              {applications.slice(0, 5).map(app => (
                <div key={app.id} className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <Building className="h-5 w-5 text-gray-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {app.id} - {app.description}
                    </p>
                    <p className="text-sm text-gray-500">{app.property}</p>
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(app.status)}`}>
                    {app.status}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const ApplicationsList = () => (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="flex-1 max-w-lg">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
            <input
              type="text"
              placeholder="Search applications..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
        
        <div className="flex gap-3">
          <select
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
          >
            <option value="all">All Statuses</option>
            {applicationStatuses.map(status => (
              <option key={status} value={status}>{status}</option>
            ))}
          </select>
          
          <button
            onClick={() => setShowNewApplicationModal(true)}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            <Plus className="h-5 w-5" />
            New Application
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Application ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Property
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Applicant
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Due Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredApplications.map((app) => {
                const daysRemaining = getDaysRemaining(app.targetDate);
                return (
                  <tr key={app.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-600">
                      {app.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {app.type}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900 max-w-xs truncate">
                      {app.property}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {app.applicant}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(app.status)}`}>
                          {app.status}
                        </span>
                        {app.systemReview && (
                          <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium flex items-center gap-1">
                            <AlertTriangle className="h-3 w-3" />
                            AI: {app.systemReview.overallScore}%
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className={`${daysRemaining < 0 ? 'text-red-600' : daysRemaining <= 7 ? 'text-orange-600' : 'text-gray-900'}`}>
                        {app.targetDate}
                        <div className="text-xs">
                          {daysRemaining < 0 ? `${Math.abs(daysRemaining)} days overdue` : 
                           daysRemaining === 0 ? 'Due today' :
                           `${daysRemaining} days left`}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setSelectedApplication(app)}
                          className="text-blue-600 hover:text-blue-900"
                          title="View Details"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button 
                          onClick={() => {
                            setSelectedApplication(app);
                            setShowSystemReviewModal(true);
                          }}
                          className="text-purple-600 hover:text-purple-900"
                          title="System Review"
                        >
                          <AlertTriangle className="h-4 w-4" />
                        </button>
                        <button 
                          onClick={() => {
                            setSelectedApplication(app);
                            setShowFormCheckerModal(true);
                          }}
                          className="text-green-600 hover:text-green-900"
                          title="Check Forms"
                        >
                          <CheckCircle className="h-4 w-4" />
                        </button>
                        <button className="text-gray-600 hover:text-gray-900" title="Edit">
                          <Edit className="h-4 w-4" />
                        </button>
                        <button className="text-red-600 hover:text-red-900" title="Delete">
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const NewApplicationModal = ({ onClose }) => {
    const [formData, setFormData] = useState({
      type: '',
      property: '',
      applicant: '',
      description: ''
    });

    const handleSubmit = () => {
      if (!formData.type || !formData.property || !formData.applicant || !formData.description) {
        addNotification('error', 'Validation Error', 'Please fill in all required fields');
        return;
      }
      
      const newId = `APP${Date.now()}`;
      const newApplication = {
        ...formData,
        id: newId,
        status: 'Draft',
        submissionDate: new Date().toISOString().split('T')[0],
        targetDate: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        assignedOfficer: 'Unassigned',
        referrals: [],
        documents: []
      };
      setApplications(prev => [...prev, newApplication]);
      addNotification('success', 'Application Created', `New application ${newId} has been created`);
      addAuditEntry('application_created', `Created new application ${newId}`, newId);
      onClose();
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-lg max-w-2xl w-full">
          <div className="p-6 border-b border-gray-200">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold text-gray-900">New Application</h2>
              <button onClick={onClose} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>
          </div>
          
          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Application Type</label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={formData.type}
                onChange={(e) => setFormData(prev => ({ ...prev, type: e.target.value }))}
              >
                <option value="">Select application type</option>
                {applicationTypes.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Property Address</label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={formData.property}
                onChange={(e) => setFormData(prev => ({ ...prev, property: e.target.value }))}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Applicant Name</label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={formData.applicant}
                onChange={(e) => setFormData(prev => ({ ...prev, applicant: e.target.value }))}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
              <textarea
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={formData.description}
                onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              />
            </div>
            
            <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
              <button
                onClick={onClose}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Create Application
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // If user is not logged in, show login modal
  if (!currentUser) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <LoginModal />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {showLoginModal && <LoginModal />}
      {showNotificationsPanel && <NotificationsPanel />}

      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Building className="h-8 w-8 text-blue-600 mr-3" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">City of Kalamunda</h1>
                <p className="text-sm text-gray-600">Building Approval System v2.0 (Production)</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">Building Act 2011 Compliant</span>
              
              {/* Notifications */}
              <div className="relative">
                <button
                  onClick={() => setShowNotificationsPanel(!showNotificationsPanel)}
                  className="relative p-2 text-gray-400 hover:text-gray-600"
                >
                  <AlertTriangle className="h-5 w-5" />
                  {notifications.filter(n => !n.read).length > 0 && (
                    <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                      {notifications.filter(n => !n.read).length}
                    </span>
                  )}
                </button>
              </div>
              
              {/* User Profile */}
              {currentUser ? (
                <div className="flex items-center space-x-3">
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">{currentUser.fullName}</p>
                    <p className="text-xs text-gray-500">{userRoles[currentUser.role].name}</p>
                  </div>
                  <Users className="h-8 w-8 text-gray-400" />
                  <button
                    onClick={logout}
                    className="text-sm text-red-600 hover:text-red-800"
                  >
                    Logout
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowLoginModal(true)}
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                >
                  Login
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8 overflow-x-auto">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                activeTab === 'dashboard'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              📊 Dashboard
            </button>
            {hasPermission('view') && (
              <button
                onClick={() => setActiveTab('applications')}
                className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  activeTab === 'applications'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                📋 Applications
              </button>
            )}
            <button
              onClick={() => setActiveTab('calendar')}
              className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                activeTab === 'calendar'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Calendar className="h-4 w-4 inline mr-1" />
              Council Calendar
            </button>
            {hasPermission('all') && (
              <button
                onClick={() => setActiveTab('analytics')}
                className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  activeTab === 'analytics'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                📈 Analytics
              </button>
            )}
          </nav>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && <DashboardView />}
        {activeTab === 'applications' && hasPermission('view') && <ApplicationsList />}
        {activeTab === 'calendar' && (
          <div className="bg-white p-8 rounded-lg shadow-sm border text-center">
            <Calendar className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Council Meeting Calendar</h3>
            <p className="text-gray-600 mb-4">Next Council meeting: Fourth Monday of each month</p>
            <p className="text-sm text-gray-500">Applications requiring Council determination will be scheduled here</p>
            
            {hasPermission('all') && (
              <div className="mt-6 space-y-2">
                <button
                  onClick={() => addNotification('info', 'Calendar Sync', 'Syncing with Outlook calendar...')}
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 mr-2"
                >
                  Sync with Outlook
                </button>
                <button
                  onClick={() => generatePDF('Council Schedule', { id: 'COUNCIL-2025', type: 'Council Schedule' })}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
                >
                  Export Schedule
                </button>
              </div>
            )}
          </div>
        )}
        {activeTab === 'analytics' && hasPermission('all') && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900">System Analytics</h2>
            
            {/* Key Performance Indicators */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <div className="flex items-center">
                  <div className="bg-blue-100 p-3 rounded-lg">
                    <FileText className="h-6 w-6 text-blue-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Monthly Applications</p>
                    <p className="text-2xl font-bold text-gray-900">127</p>
                    <p className="text-xs text-green-600">+12% from last month</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <div className="flex items-center">
                  <div className="bg-green-100 p-3 rounded-lg">
                    <CheckCircle className="h-6 w-6 text-green-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Approval Rate</p>
                    <p className="text-2xl font-bold text-gray-900">89.3%</p>
                    <p className="text-xs text-green-600">+2.1% from last month</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <div className="flex items-center">
                  <div className="bg-yellow-100 p-3 rounded-lg">
                    <Clock className="h-6 w-6 text-yellow-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Avg. Processing</p>
                    <p className="text-2xl font-bold text-gray-900">3.8 days</p>
                    <p className="text-xs text-green-600">-0.4 days improvement</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <div className="flex items-center">
                  <div className="bg-purple-100 p-3 rounded-lg">
                    <Users className="h-6 w-6 text-purple-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">User Satisfaction</p>
                    <p className="text-2xl font-bold text-gray-900">4.7/5</p>
                    <p className="text-xs text-green-600">+0.2 improvement</p>
                  </div>
                </div>
              </div>
            </div>
            
            {/* System Health */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Server Uptime</span>
                    <span className="text-sm font-medium text-green-600">99.94%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Database Performance</span>
                    <span className="text-sm font-medium text-green-600">Excellent</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">API Response Time</span>
                    <span className="text-sm font-medium text-yellow-600">124ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Active Users</span>
                    <span className="text-sm font-medium text-blue-600">23</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Integration Status</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">WA Planning Portal</span>
                    <span className="text-sm font-medium text-green-600">Connected</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Main Roads WA</span>
                    <span className="text-sm font-medium text-green-600">Connected</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Water Corporation</span>
                    <span className="text-sm font-medium text-yellow-600">Limited</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Western Power</span>
                    <span className="text-sm font-medium text-green-600">Connected</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Recent Activity Summary */}
            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity Summary</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">47</div>
                  <div className="text-sm text-gray-600">Applications This Week</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600">34</div>
                  <div className="text-sm text-gray-600">Approvals This Week</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-orange-600">8</div>
                  <div className="text-sm text-gray-600">Pending Reviews</div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Unauthorized access message */}
        {!hasPermission('view') && ['applications'].includes(activeTab) && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-red-900 mb-2">Access Denied</h3>
            <p className="text-red-700">You don't have permission to access this section.</p>
          </div>
        )}
        
        {!hasPermission('all') && ['analytics'].includes(activeTab) && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-red-900 mb-2">Administrator Access Required</h3>
            <p className="text-red-700">This section is only accessible to system administrators.</p>
          </div>
        )}
      </div>

      {showNewApplicationModal && (
        <NewApplicationModal onClose={() => setShowNewApplicationModal(false)} />
      )}

      {/* Production Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4">City of Kalamunda</h3>
              <p className="text-gray-300 text-sm">
                Building Approval System v2.0
                <br />
                Production Environment
                <br />
                Building Act 2011 Compliant
              </p>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">System Status</h3>
              <div className="text-sm text-gray-300 space-y-1">
                <p>🟢 All Systems Operational</p>
                <p>📊 Uptime: {performanceMetrics.systemUptime}</p>
                <p>🔒 Security: Enterprise Grade</p>
                <p>🌐 API Status: Healthy</p>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">Contact Support</h3>
              <div className="text-sm text-gray-300 space-y-1">
                <p>📞 (08) 9257 9999</p>
                <p>📧 itsupport@kalamunda.wa.gov.au</p>
                <p>🕒 Mon-Fri 8:00 AM - 5:00 PM</p>
                <p>🏢 2 Railway Road, Kalamunda</p>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
              <div className="text-sm text-gray-300 space-y-1">
                <p>📋 User Manual</p>
                <p>🔧 System Requirements</p>
                <p>🔐 Privacy Policy</p>
                <p>📊 Service Status</p>
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-700 mt-8 pt-8 text-center text-sm text-gray-400">
            <p>© 2025 City of Kalamunda. All rights reserved. | System administered by IT Department</p>
            <p className="mt-2">
              Last System Update: June 21, 2025 | Database Version: 2.1.4 | 
              {currentUser && ` Logged in as: ${currentUser.fullName} (${userRoles[currentUser.role].name})`}
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default BuildingApprovalSystem;
