import React, { useState, useEffect } from 'react';
import { Calendar, FileText, Clock, CheckCircle, AlertTriangle, Users, Building, Search, Plus, Eye, Edit, Trash2, Download } from 'lucide-react';

const BuildingApprovalSystem = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [applications, setApplications] = useState([]);
  const [selectedApplication, setSelectedApplication] = useState(null);
  const [showNewApplicationModal, setShowNewApplicationModal] = useState(false);
  const [showFormCheckerModal, setShowFormCheckerModal] = useState(false);
  const [showSystemReviewModal, setShowSystemReviewModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');

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

  // Site plan validation requirements by zone
  const sitePlanRequirements = {
    'Residential R20': {
      frontSetback: 6.0,
      sideSetback: 1.5,
      rearSetback: 6.0,
      maxBuildingHeight: 9.0,
      maxPlotRatio: 0.5,
      minLandscaping: 50,
      minParkingSpaces: 2,
      maxBuildingCoverage: 50
    },
    'Residential R40': {
      frontSetback: 4.0,
      sideSetback: 1.2,
      rearSetback: 4.0,
      maxBuildingHeight: 9.0,
      maxPlotRatio: 0.6,
      minLandscaping: 40,
      minParkingSpaces: 1.5,
      maxBuildingCoverage: 60
    },
    'Commercial': {
      frontSetback: 3.0,
      sideSetback: 0,
      rearSetback: 3.0,
      maxBuildingHeight: 12.0,
      maxPlotRatio: 1.0,
      minLandscaping: 20,
      minParkingSpaces: 3.5,
      maxBuildingCoverage: 80
    },
    'Industrial': {
      frontSetback: 10.0,
      sideSetback: 3.0,
      rearSetback: 6.0,
      maxBuildingHeight: 15.0,
      maxPlotRatio: 0.8,
      minLandscaping: 15,
      minParkingSpaces: 2.0,
      maxBuildingCoverage: 70
    }
  };

  // Simulate site plan analysis
  const analyzeSitePlan = (application) => {
    // Simulate extraction of site plan data (in real system would use AI/OCR)
    const mockSitePlanData = {
      lotArea: Math.floor(Math.random() * 500) + 400, // 400-900 sqm
      buildingArea: Math.floor(Math.random() * 200) + 150, // 150-350 sqm
      frontSetback: Math.random() * 8 + 2, // 2-10m
      sideSetback: Math.random() * 3 + 0.5, // 0.5-3.5m
      rearSetback: Math.random() * 8 + 2, // 2-10m
      buildingHeight: Math.random() * 12 + 6, // 6-18m
      parkingSpaces: Math.floor(Math.random() * 4) + 1, // 1-4 spaces
      landscapedArea: Math.floor(Math.random() * 40) + 20, // 20-60%
      zoning: ['Residential R20', 'Residential R40', 'Commercial', 'Industrial'][Math.floor(Math.random() * 4)],
      hasStormwater: Math.random() > 0.3,
      hasServices: Math.random() > 0.2,
      neighborConsultation: Math.random() > 0.4
    };

    const requirements = sitePlanRequirements[mockSitePlanData.zoning];
    const violations = [];
    const warnings = [];
    const compliant = [];

    // Check setbacks
    if (mockSitePlanData.frontSetback < requirements.frontSetback) {
      violations.push(`Front setback ${mockSitePlanData.frontSetback.toFixed(1)}m < required ${requirements.frontSetback}m`);
    } else {
      compliant.push(`Front setback compliant: ${mockSitePlanData.frontSetback.toFixed(1)}m`);
    }

    if (mockSitePlanData.sideSetback < requirements.sideSetback) {
      violations.push(`Side setback ${mockSitePlanData.sideSetback.toFixed(1)}m < required ${requirements.sideSetback}m`);
    } else {
      compliant.push(`Side setback compliant: ${mockSitePlanData.sideSetback.toFixed(1)}m`);
    }

    if (mockSitePlanData.rearSetback < requirements.rearSetback) {
      violations.push(`Rear setback ${mockSitePlanData.rearSetback.toFixed(1)}m < required ${requirements.rearSetback}m`);
    } else {
      compliant.push(`Rear setback compliant: ${mockSitePlanData.rearSetback.toFixed(1)}m`);
    }

    // Check building coverage
    const buildingCoverage = (mockSitePlanData.buildingArea / mockSitePlanData.lotArea) * 100;
    if (buildingCoverage > requirements.maxBuildingCoverage) {
      violations.push(`Building coverage ${buildingCoverage.toFixed(1)}% > maximum ${requirements.maxBuildingCoverage}%`);
    } else {
      compliant.push(`Building coverage compliant: ${buildingCoverage.toFixed(1)}%`);
    }

    // Check plot ratio
    const plotRatio = mockSitePlanData.buildingArea / mockSitePlanData.lotArea;
    if (plotRatio > requirements.maxPlotRatio) {
      violations.push(`Plot ratio ${plotRatio.toFixed(2)} > maximum ${requirements.maxPlotRatio}`);
    } else {
      compliant.push(`Plot ratio compliant: ${plotRatio.toFixed(2)}`);
    }

    // Check building height
    if (mockSitePlanData.buildingHeight > requirements.maxBuildingHeight) {
      violations.push(`Building height ${mockSitePlanData.buildingHeight.toFixed(1)}m > maximum ${requirements.maxBuildingHeight}m`);
    } else {
      compliant.push(`Building height compliant: ${mockSitePlanData.buildingHeight.toFixed(1)}m`);
    }

    // Check parking
    if (mockSitePlanData.parkingSpaces < requirements.minParkingSpaces) {
      violations.push(`Parking ${mockSitePlanData.parkingSpaces} spaces < required ${requirements.minParkingSpaces}`);
    } else {
      compliant.push(`Parking provision compliant: ${mockSitePlanData.parkingSpaces} spaces`);
    }

    // Check landscaping
    if (mockSitePlanData.landscapedArea < requirements.minLandscaping) {
      violations.push(`Landscaping ${mockSitePlanData.landscapedArea}% < required ${requirements.minLandscaping}%`);
    } else {
      compliant.push(`Landscaping compliant: ${mockSitePlanData.landscapedArea}%`);
    }

    // Check services and infrastructure
    if (!mockSitePlanData.hasStormwater) {
      warnings.push('Stormwater management details not clear on site plan');
    }

    if (!mockSitePlanData.hasServices) {
      warnings.push('Utility services connection not shown');
    }

    if (!mockSitePlanData.neighborConsultation && violations.length > 0) {
      warnings.push('Neighbor consultation may be required due to non-compliances');
    }

    // Calculate compliance score
    const totalChecks = 8;
    const compliantChecks = compliant.length;
    const complianceScore = Math.round((compliantChecks / totalChecks) * 100);

    return {
      sitePlanData: mockSitePlanData,
      requirements,
      violations,
      warnings,
      compliant,
      complianceScore,
      recommendation: complianceScore >= 90 ? 'Approve site plan' :
                     complianceScore >= 70 ? 'Approve with conditions' :
                     complianceScore >= 50 ? 'Request modifications' : 'Reject - major non-compliance'
    };
  };

  // System review automation functions
  const performSystemReview = (application) => {
    const requiredDocs = requiredDocuments[application.type] || [];
    const submittedDocs = application.documents || [];
    
    // Document completeness analysis
    const documentScore = (submittedDocs.length / requiredDocs.length) * 100;
    const missingDocs = requiredDocs.filter(doc => !submittedDocs.includes(doc));
    
    // Site plan analysis (if site plan is submitted)
    let sitePlanAnalysis = null;
    const hasSitePlan = submittedDocs.some(doc => 
      doc.toLowerCase().includes('site plan') || 
      doc.toLowerCase().includes('plans') ||
      doc.toLowerCase().includes('site analysis')
    );
    
    if (hasSitePlan) {
      sitePlanAnalysis = analyzeSitePlan(application);
    }
    
    // Application completeness analysis
    const requiredFields = ['type', 'property', 'applicant', 'description'];
    const fieldCompleteness = requiredFields.every(field => application[field]) ? 100 : 80;
    
    // Risk assessment based on application type and value
    const riskFactors = {
      'Development Approval': 3,
      'Building Permit - Certified (BA01)': 2,
      'Building Permit - Uncertified (BA02)': 4,
      'Demolition Permit (BA05)': 5,
      'Occupancy Permit (BA09)': 2,
      'Building Approval Certificate (BA13)': 4
    };
    const riskLevel = riskFactors[application.type] || 3;
    
    // Compliance predictions
    const complianceIssues = [];
    if (missingDocs.length > 0) {
      complianceIssues.push(`Missing ${missingDocs.length} required documents`);
    }
    if (application.type.includes('Certified') && !submittedDocs.includes('Certificate of Design Compliance (BA03)')) {
      complianceIssues.push('Professional certification required for certified applications');
    }
    if (application.publicNotification && !application.referrals?.includes('Public')) {
      complianceIssues.push('Public notification process not initiated');
    }
    
    // Add site plan issues
    if (sitePlanAnalysis) {
      if (sitePlanAnalysis.violations.length > 0) {
        complianceIssues.push(`${sitePlanAnalysis.violations.length} site plan violations detected`);
      }
      if (sitePlanAnalysis.warnings.length > 0) {
        complianceIssues.push(`${sitePlanAnalysis.warnings.length} site plan warnings identified`);
      }
    } else if (['Development Approval', 'Building Permit - Certified (BA01)', 'Building Permit - Uncertified (BA02)'].includes(application.type)) {
      complianceIssues.push('Site plan required but not submitted or not analyzable');
    }
    
    // Overall system score (incorporating site plan)
    let baseScore = (documentScore + fieldCompleteness - (riskLevel * 5)) / 2;
    if (sitePlanAnalysis) {
      baseScore = (baseScore + sitePlanAnalysis.complianceScore) / 2;
    }
    const overallScore = Math.max(0, Math.min(100, Math.round(baseScore)));
    
    // System recommendations
    const systemRecommendations = [];
    if (overallScore >= 85) {
      systemRecommendations.push('Application appears ready for approval - recommend expedited review');
    } else if (overallScore >= 70) {
      systemRecommendations.push('Application meets basic requirements - standard review recommended');
    } else if (overallScore >= 50) {
      systemRecommendations.push('Application has deficiencies - request additional information');
    } else {
      systemRecommendations.push('Application incomplete - return to applicant');
    }
    
    // Add site plan specific recommendations
    if (sitePlanAnalysis) {
      systemRecommendations.push(`Site plan analysis: ${sitePlanAnalysis.recommendation}`);
    }
    
    // Auto-populate checklist (including site plan items)
    const autoChecklist = getFormChecklist(application.type).map(item => {
      let checked = false;
      let comments = '';
      
      if (item.item.includes('documents')) {
        checked = missingDocs.length === 0;
        if (missingDocs.length > 0) {
          comments = `Missing: ${missingDocs.slice(0, 2).join(', ')}${missingDocs.length > 2 ? '...' : ''}`;
        }
      } else if (item.item.includes('form completed')) {
        checked = fieldCompleteness === 100;
      } else if (item.item.includes('Building Services Levy')) {
        checked = Math.random() > 0.3;
      } else if (item.item.includes('address verified')) {
        checked = true;
      } else if (item.item.includes('applicant details')) {
        checked = fieldCompleteness === 100;
      } else if (item.item.includes('Professional certifications')) {
        checked = !application.type.includes('Certified') || submittedDocs.includes('Certificate of Design Compliance (BA03)');
        if (application.type.includes('Certified')) {
          comments = 'Certified application - verify BA03 certificate';
        }
      } else {
        checked = Math.random() > 0.4;
      }
      
      return { ...item, checked, comments };
    });
    
    // Add site plan specific checklist items
    if (sitePlanAnalysis) {
      autoChecklist.push({
        item: 'Site plan setbacks compliance',
        required: true,
        checked: sitePlanAnalysis.violations.filter(v => v.includes('setback')).length === 0,
        comments: sitePlanAnalysis.violations.filter(v => v.includes('setback')).join('; ')
      });
      
      autoChecklist.push({
        item: 'Site plan building coverage compliance',
        required: true,
        checked: !sitePlanAnalysis.violations.some(v => v.includes('coverage')),
        comments: sitePlanAnalysis.violations.find(v => v.includes('coverage')) || ''
      });
      
      autoChecklist.push({
        item: 'Site plan parking provision',
        required: true,
        checked: !sitePlanAnalysis.violations.some(v => v.includes('Parking')),
        comments: sitePlanAnalysis.violations.find(v => v.includes('Parking')) || ''
      });
    }
    
    return {
      documentScore,
      fieldCompleteness,
      riskLevel,
      complianceIssues,
      overallScore,
      systemRecommendations,
      missingDocs,
      autoChecklist,
      sitePlanAnalysis,
      reviewDate: new Date().toISOString().split('T')[0],
      processingTime: Math.floor(Math.random() * 3) + 1 // 1-3 seconds simulation
    };
  };

  const SystemReviewModal = ({ application, onClose, onApplyToFormChecker }) => {
    const [isAnalyzing, setIsAnalyzing] = useState(true);
    const [systemResults, setSystemResults] = useState(null);

    useEffect(() => {
      // Simulate system analysis time (2-4 seconds)
      const processingTime = Math.floor(Math.random() * 3) + 2;
      const timer = setTimeout(() => {
        const results = performSystemReview(application);
        setSystemResults(results);
        setIsAnalyzing(false);
      }, processingTime * 1000);

      return () => clearTimeout(timer);
    }, [application]);

    const getScoreColor = (score) => {
      if (score >= 85) return 'text-green-600 bg-green-100';
      if (score >= 70) return 'text-blue-600 bg-blue-100';
      if (score >= 50) return 'text-yellow-600 bg-yellow-100';
      return 'text-red-600 bg-red-100';
    };

    const getRiskColor = (level) => {
      if (level <= 2) return 'text-green-600 bg-green-100';
      if (level <= 3) return 'text-yellow-600 bg-yellow-100';
      return 'text-red-600 bg-red-100';
    };

    if (isAnalyzing) {
      return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full p-8 text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">AI System Analysis in Progress</h2>
            <p className="text-gray-600 mb-4">Analyzing application compliance, site plan requirements, and generating recommendations...</p>
            <div className="space-y-2 text-sm text-gray-500">
              <p>✓ Checking document completeness</p>
              <p>✓ Validating form data integrity</p>
              <p>✓ Analyzing site plan compliance</p>
              <p>✓ Assessing planning requirements</p>
              <p>✓ Generating recommendations</p>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-y-auto">
          <div className="p-6 border-b border-gray-200">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold text-gray-900">
                System Review Results - {application.id}
              </h2>
              <button onClick={onClose} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>
          </div>
          
          <div className="p-6 space-y-6">
            {/* System Score Dashboard */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div className={`p-4 rounded-lg ${getScoreColor(systemResults.overallScore)}`}>
                <div className="text-center">
                  <div className="text-2xl font-bold">{systemResults.overallScore}%</div>
                  <div className="text-sm font-medium">Overall Score</div>
                </div>
              </div>
              <div className={`p-4 rounded-lg ${getScoreColor(systemResults.documentScore)}`}>
                <div className="text-center">
                  <div className="text-2xl font-bold">{Math.round(systemResults.documentScore)}%</div>
                  <div className="text-sm font-medium">Document Complete</div>
                </div>
              </div>
              <div className={`p-4 rounded-lg ${getScoreColor(systemResults.fieldCompleteness)}`}>
                <div className="text-center">
                  <div className="text-2xl font-bold">{systemResults.fieldCompleteness}%</div>
                  <div className="text-sm font-medium">Form Complete</div>
                </div>
              </div>
              {systemResults.sitePlanAnalysis && (
                <div className={`p-4 rounded-lg ${getScoreColor(systemResults.sitePlanAnalysis.complianceScore)}`}>
                  <div className="text-center">
                    <div className="text-2xl font-bold">{systemResults.sitePlanAnalysis.complianceScore}%</div>
                    <div className="text-sm font-medium">Site Plan</div>
                  </div>
                </div>
              )}
              <div className={`p-4 rounded-lg ${getRiskColor(systemResults.riskLevel)}`}>
                <div className="text-center">
                  <div className="text-2xl font-bold">{systemResults.riskLevel}/5</div>
                  <div className="text-sm font-medium">Risk Level</div>
                </div>
              </div>
            </div>

            {/* System Recommendations */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-semibold text-blue-900 mb-3 flex items-center gap-2">
                <CheckCircle className="h-5 w-5" />
                System Recommendations
              </h3>
              <ul className="space-y-2">
                {systemResults.systemRecommendations.map((rec, index) => (
                  <li key={index} className="text-blue-800 text-sm flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>

            {/* Compliance Issues */}
            {systemResults.complianceIssues.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <h3 className="font-semibold text-red-900 mb-3 flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Compliance Issues Detected
                </h3>
                <ul className="space-y-2">
                  {systemResults.complianceIssues.map((issue, index) => (
                    <li key={index} className="text-red-800 text-sm flex items-start gap-2">
                      <span className="text-red-600 mt-1">•</span>
                      {issue}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Site Plan Analysis Results */}
            {systemResults.sitePlanAnalysis && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center gap-2">
                  🗺️ Site Plan Analysis Results
                </h3>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Site Plan Metrics */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-900 mb-3">Site Plan Metrics</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Zoning:</span>
                        <span className="font-medium">{systemResults.sitePlanAnalysis.sitePlanData.zoning}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Lot Area:</span>
                        <span className="font-medium">{systemResults.sitePlanAnalysis.sitePlanData.lotArea}m²</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Building Area:</span>
                        <span className="font-medium">{systemResults.sitePlanAnalysis.sitePlanData.buildingArea}m²</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Building Coverage:</span>
                        <span className="font-medium">
                          {((systemResults.sitePlanAnalysis.sitePlanData.buildingArea / systemResults.sitePlanAnalysis.sitePlanData.lotArea) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Building Height:</span>
                        <span className="font-medium">{systemResults.sitePlanAnalysis.sitePlanData.buildingHeight.toFixed(1)}m</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Parking Spaces:</span>
                        <span className="font-medium">{systemResults.sitePlanAnalysis.sitePlanData.parkingSpaces}</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Setback Analysis */}
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-3">Setback Analysis</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Front Setback:</span>
                        <div className="text-right">
                          <div className={`text-sm font-medium ${
                            systemResults.sitePlanAnalysis.sitePlanData.frontSetback >= systemResults.sitePlanAnalysis.requirements.frontSetback 
                              ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {systemResults.sitePlanAnalysis.sitePlanData.frontSetback.toFixed(1)}m
                          </div>
                          <div className="text-xs text-gray-500">
                            Req: {systemResults.sitePlanAnalysis.requirements.frontSetback}m
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Side Setback:</span>
                        <div className="text-right">
                          <div className={`text-sm font-medium ${
                            systemResults.sitePlanAnalysis.sitePlanData.sideSetback >= systemResults.sitePlanAnalysis.requirements.sideSetback 
                              ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {systemResults.sitePlanAnalysis.sitePlanData.sideSetback.toFixed(1)}m
                          </div>
                          <div className="text-xs text-gray-500">
                            Req: {systemResults.sitePlanAnalysis.requirements.sideSetback}m
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Rear Setback:</span>
                        <div className="text-right">
                          <div className={`text-sm font-medium ${
                            systemResults.sitePlanAnalysis.sitePlanData.rearSetback >= systemResults.sitePlanAnalysis.requirements.rearSetback 
                              ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {systemResults.sitePlanAnalysis.sitePlanData.rearSetback.toFixed(1)}m
                          </div>
                          <div className="text-xs text-gray-500">
                            Req: {systemResults.sitePlanAnalysis.requirements.rearSetback}m
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Compliance Status */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                  {/* Violations */}
                  {systemResults.sitePlanAnalysis.violations.length > 0 && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <h4 className="font-semibold text-red-900 mb-2 flex items-center gap-1">
                        <AlertTriangle className="h-4 w-4" />
                        Violations ({systemResults.sitePlanAnalysis.violations.length})
                      </h4>
                      <ul className="text-sm text-red-800 space-y-1">
                        {systemResults.sitePlanAnalysis.violations.slice(0, 3).map((violation, index) => (
                          <li key={index} className="flex items-start gap-1">
                            <span className="text-red-600 mt-0.5">•</span>
                            {violation}
                          </li>
                        ))}
                        {systemResults.sitePlanAnalysis.violations.length > 3 && (
                          <li className="text-red-600 text-xs">
                            +{systemResults.sitePlanAnalysis.violations.length - 3} more...
                          </li>
                        )}
                      </ul>
                    </div>
                  )}
                  
                  {/* Warnings */}
                  {systemResults.sitePlanAnalysis.warnings.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <h4 className="font-semibold text-yellow-900 mb-2 flex items-center gap-1">
                        <AlertTriangle className="h-4 w-4" />
                        Warnings ({systemResults.sitePlanAnalysis.warnings.length})
                      </h4>
                      <ul className="text-sm text-yellow-800 space-y-1">
                        {systemResults.sitePlanAnalysis.warnings.map((warning, index) => (
                          <li key={index} className="flex items-start gap-1">
                            <span className="text-yellow-600 mt-0.5">•</span>
                            {warning}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* Compliant Items */}
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h4 className="font-semibold text-green-900 mb-2 flex items-center gap-1">
                      <CheckCircle className="h-4 w-4" />
                      Compliant ({systemResults.sitePlanAnalysis.compliant.length})
                    </h4>
                    <ul className="text-sm text-green-800 space-y-1">
                      {systemResults.sitePlanAnalysis.compliant.slice(0, 3).map((item, index) => (
                        <li key={index} className="flex items-start gap-1">
                          <span className="text-green-600 mt-0.5">•</span>
                          {item}
                        </li>
                      ))}
                      {systemResults.sitePlanAnalysis.compliant.length > 3 && (
                        <li className="text-green-600 text-xs">
                          +{systemResults.sitePlanAnalysis.compliant.length - 3} more...
                        </li>
                      )}
                    </ul>
                  </div>
                </div>
                
                {/* Site Plan Recommendation */}
                <div className={`mt-4 p-4 rounded-lg border ${
                  systemResults.sitePlanAnalysis.complianceScore >= 90 ? 'bg-green-50 border-green-200' :
                  systemResults.sitePlanAnalysis.complianceScore >= 70 ? 'bg-blue-50 border-blue-200' :
                  systemResults.sitePlanAnalysis.complianceScore >= 50 ? 'bg-yellow-50 border-yellow-200' : 
                  'bg-red-50 border-red-200'
                }`}>
                  <div className="flex items-center gap-3">
                    <div className={`text-2xl font-bold ${
                      systemResults.sitePlanAnalysis.complianceScore >= 90 ? 'text-green-600' :
                      systemResults.sitePlanAnalysis.complianceScore >= 70 ? 'text-blue-600' :
                      systemResults.sitePlanAnalysis.complianceScore >= 50 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {systemResults.sitePlanAnalysis.complianceScore}%
                    </div>
                    <div className="flex-1">
                      <div className="font-semibold text-gray-900">Site Plan Compliance Score</div>
                      <div className={`text-sm font-medium ${
                        systemResults.sitePlanAnalysis.complianceScore >= 90 ? 'text-green-700' :
                        systemResults.sitePlanAnalysis.complianceScore >= 70 ? 'text-blue-700' :
                        systemResults.sitePlanAnalysis.complianceScore >= 50 ? 'text-yellow-700' : 'text-red-700'
                      }`}>
                        Recommendation: {systemResults.sitePlanAnalysis.recommendation}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Document Analysis */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Document Status</h3>
                <div className="space-y-2">
                  {requiredDocuments[application.type]?.map((doc, index) => {
                    const isSubmitted = application.documents.includes(doc);
                    return (
                      <div key={index} className="flex items-center gap-2 text-sm">
                        <div className={`w-3 h-3 rounded-full ${isSubmitted ? 'bg-green-500' : 'bg-red-500'}`} />
                        <span className={isSubmitted ? 'text-gray-900' : 'text-red-600'}>
                          {doc}
                        </span>
                        {!isSubmitted && <span className="text-red-500 text-xs">(Missing)</span>}
                      </div>
                    );
                  })}
                </div>
              </div>

              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Auto-Generated Checklist Preview</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {systemResults.autoChecklist.slice(0, 6).map((item, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={item.checked}
                        readOnly
                        className="h-3 w-3 text-blue-600 rounded"
                      />
                      <span className={`flex-1 ${item.checked ? 'text-green-700' : 'text-gray-700'}`}>
                        {item.item}
                      </span>
                      {item.required && <span className="text-red-500 text-xs">*</span>}
                    </div>
                  ))}
                  {systemResults.autoChecklist.length > 6 && (
                    <p className="text-gray-500 text-xs">+ {systemResults.autoChecklist.length - 6} more items...</p>
                  )}
                </div>
              </div>
            </div>

            {/* Processing Summary */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">System Analysis Summary</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-700">Analysis Date:</span>
                  <span className="ml-2 text-gray-900">{systemResults.reviewDate}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Processing Time:</span>
                  <span className="ml-2 text-gray-900">{systemResults.processingTime} seconds</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Documents Checked:</span>
                  <span className="ml-2 text-gray-900">{application.documents.length} of {requiredDocuments[application.type]?.length || 0}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Auto-Completion Rate:</span>
                  <span className="ml-2 text-gray-900">{Math.round((systemResults.autoChecklist.filter(item => item.checked).length / systemResults.autoChecklist.length) * 100)}%</span>
                </div>
                {systemResults.sitePlanAnalysis && (
                  <>
                    <div>
                      <span className="font-medium text-gray-700">Site Plan Analyzed:</span>
                      <span className="ml-2 text-green-600 font-medium">✓ Yes</span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Planning Violations:</span>
                      <span className={`ml-2 font-medium ${systemResults.sitePlanAnalysis.violations.length > 0 ? 'text-red-600' : 'text-green-600'}`}>
                        {systemResults.sitePlanAnalysis.violations.length}
                      </span>
                    </div>
                  </>
                )}
                {!systemResults.sitePlanAnalysis && ['Development Approval', 'Building Permit - Certified (BA01)', 'Building Permit - Uncertified (BA02)'].includes(application.type) && (
                  <div className="col-span-2">
                    <span className="font-medium text-gray-700">Site Plan Status:</span>
                    <span className="ml-2 text-orange-600 font-medium">⚠️ Not analyzed (plan not found or unreadable)</span>
                  </div>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-between items-center pt-4 border-t border-gray-200">
              <div className="text-sm text-gray-600">
                <span className="flex items-center gap-1">
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  System analysis complete - ready for officer review
                </span>
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={onClose}
                  className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                >
                  Close
                </button>
                <button
                  onClick={() => {
                    // Save system results to application
                    setApplications(prev => prev.map(app => 
                      app.id === application.id 
                        ? { ...app, systemReview: systemResults }
                        : app
                    ));
                    onClose();
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Save Results
                </button>
                <button
                  onClick={() => onApplyToFormChecker(systemResults)}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
                >
                  <CheckCircle className="h-4 w-4" />
                  Apply to Form Checker
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const getFormChecklist = (applicationType) => [
    { item: 'Application form completed correctly', required: true },
    { item: 'All required documents submitted', required: true },
    { item: 'Building Services Levy paid', required: true },
    { item: 'Site address verified', required: true },
    { item: 'Applicant details complete', required: true },
    { item: 'Professional certifications (if required)', required: applicationType.includes('Certified') },
    { item: 'Public notification completed (if required)', required: false },
    { item: 'Referral responses received', required: false },
    { item: 'Compliance with planning scheme', required: true },
    { item: 'Building Code compliance verified', required: true }
  ];

  useEffect(() => {
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
        documents: ['Site Plan', 'Floor Plans', 'Elevations', 'Landscape Plan', 'Traffic Report']
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
        documents: ['Certified Plans', 'Site Plan', 'Structural Report', 'Fire Safety Plan', 'Certificate of Design Compliance (BA03)']
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
        documents: ['Architectural Plans', 'Site Plan', 'Site Analysis', 'Energy Report']
      }
    ];
    setApplications(sampleApplications);
  }, []);

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

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
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
            <h3 className="text-lg font-semibold text-gray-900 mb-4">System Analysis</h3>
            <div className="space-y-4">
              {applications.filter(app => app.systemReview).map(app => (
                <div key={app.id} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{app.id}</p>
                    <p className="text-xs text-gray-500">{app.type}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-bold ${
                      app.systemReview.overallScore >= 85 ? 'text-green-600' :
                      app.systemReview.overallScore >= 70 ? 'text-blue-600' :
                      app.systemReview.overallScore >= 50 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {app.systemReview.overallScore}%
                    </div>
                    <p className="text-xs text-gray-500">AI Score</p>
                  </div>
                </div>
              ))}
              {applications.filter(app => app.systemReview).length === 0 && (
                <p className="text-sm text-gray-500 text-center py-4">No system reviews yet</p>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Site Plan Analysis</h3>
            <div className="space-y-4">
              {applications.filter(app => app.systemReview?.sitePlanAnalysis).map(app => (
                <div key={app.id} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{app.id}</p>
                    <p className="text-xs text-gray-500">{app.systemReview.sitePlanAnalysis.sitePlanData.zoning}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-bold flex items-center gap-1 ${
                      app.systemReview.sitePlanAnalysis.complianceScore >= 85 ? 'text-green-600' :
                      app.systemReview.sitePlanAnalysis.complianceScore >= 70 ? 'text-blue-600' :
                      app.systemReview.sitePlanAnalysis.complianceScore >= 50 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      🗺️ {app.systemReview.sitePlanAnalysis.complianceScore}%
                    </div>
                    <p className="text-xs text-gray-500">
                      {app.systemReview.sitePlanAnalysis.violations.length} violations
                    </p>
                  </div>
                </div>
              ))}
              {applications.filter(app => app.systemReview?.sitePlanAnalysis).length === 0 && (
                <p className="text-sm text-gray-500 text-center py-4">No site plan analyses yet</p>
              )}
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

  const ReviewQueue = () => {
    const reviewableStatuses = ['DCU Review', 'Internal Referral', 'External Referral', 'Officer Assessment', 'Awaiting Council'];
    const applicationsForReview = applications.filter(app => reviewableStatuses.includes(app.status));
    
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-900">Review Queue</h2>
          <div className="text-sm text-gray-600">
            {applicationsForReview.length} applications requiring review
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {applicationsForReview.map(app => {
            const daysRemaining = getDaysRemaining(app.targetDate);
            const isUrgent = daysRemaining <= 7;
            const isOverdue = daysRemaining < 0;
            
            return (
              <div key={app.id} className={`bg-white rounded-lg shadow-sm border-l-4 p-6 ${
                isOverdue ? 'border-red-500' : isUrgent ? 'border-orange-500' : 'border-blue-500'
              }`}>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-blue-600">{app.id}</h3>
                    <p className="text-sm text-gray-600">{app.type}</p>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(app.status)}`}>
                    {app.status}
                  </span>
                </div>
                
                <div className="space-y-2 mb-4">
                  <div>
                    <span className="text-sm font-medium text-gray-700">Property:</span>
                    <p className="text-sm text-gray-900 truncate">{app.property}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-700">Applicant:</span>
                    <p className="text-sm text-gray-900">{app.applicant}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-700">Officer:</span>
                    <p className="text-sm text-gray-900">{app.assignedOfficer}</p>
                  </div>
                </div>
                
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <span className="text-sm font-medium text-gray-700">Due Date:</span>
                    <p className={`text-sm ${isOverdue ? 'text-red-600' : isUrgent ? 'text-orange-600' : 'text-gray-900'}`}>
                      {app.targetDate}
                    </p>
                    <p className={`text-xs ${isOverdue ? 'text-red-600' : isUrgent ? 'text-orange-600' : 'text-gray-600'}`}>
                      {isOverdue ? `${Math.abs(daysRemaining)} days overdue` : 
                       daysRemaining === 0 ? 'Due today' : `${daysRemaining} days left`}
                    </p>
                  </div>
                  
                  {isOverdue && (
                    <AlertTriangle className="h-5 w-5 text-red-500" />
                  )}
                  {isUrgent && !isOverdue && (
                    <Clock className="h-5 w-5 text-orange-500" />
                  )}
                </div>
                
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setSelectedApplication(app);
                      setShowSystemReviewModal(true);
                    }}
                    className="flex-1 bg-purple-600 text-white px-3 py-2 rounded text-sm hover:bg-purple-700 flex items-center justify-center gap-1"
                  >
                    <AlertTriangle className="h-4 w-4" />
                    System Review
                  </button>
                  <button
                    onClick={() => {
                      setSelectedApplication(app);
                      setShowFormCheckerModal(true);
                    }}
                    className="flex-1 bg-green-600 text-white px-3 py-2 rounded text-sm hover:bg-green-700 flex items-center justify-center gap-1"
                  >
                    <CheckCircle className="h-4 w-4" />
                    Officer Review
                  </button>
                </div>
                
                {app.referrals && app.referrals.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <span className="text-xs font-medium text-gray-700">Referrals:</span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {app.referrals.map((referral, index) => (
                        <span key={index} className="inline-block bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded">
                          {referral}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        {applicationsForReview.length === 0 && (
          <div className="text-center py-12">
            <CheckCircle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Applications for Review</h3>
            <p className="text-gray-600">All applications are up to date!</p>
          </div>
        )}
      </div>
    );
  };

  const FormsReference = () => {
    const forms = [
      { code: 'BA01', name: 'Building permit - certified', fee: '0.19% (Class 1/10) or 0.09% (Class 2-9), min $110' },
      { code: 'BA02', name: 'Building permit - uncertified', fee: '0.32% of work value, min $110' },
      { code: 'BA05', name: 'Demolition permit', fee: '$110 (Class 1/10) or $110 per storey (Class 2-9)' },
      { code: 'BA09', name: 'Occupancy permit', fee: '$110' },
      { code: 'BA13', name: 'Building approval certificate', fee: '0.38% of work value, min $110' },
      { code: 'BA07', name: 'Notice of completion', fee: 'No fee' },
      { code: 'BA19', name: 'Amend building permit/builder details', fee: 'Varies' },
      { code: 'BA22', name: 'Extend building/demolition permit', fee: '$110' }
    ];

    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Building Application Forms Reference</h2>
          <p className="text-gray-600 mb-6">
            Complete guide to building application forms as required under the Building Act 2011.
          </p>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-900 mb-2">Building Services Levy</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Work over $45,000: 0.137% of value</li>
                <li>• Work $45,000 or less: $61.65 flat fee</li>
                <li>• Unauthorized work: 0.274% of value</li>
              </ul>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="font-semibold text-green-900 mb-2">Contact</h3>
              <ul className="text-sm text-green-800 space-y-1">
                <li>• Phone: (08) 9257 9999</li>
                <li>• Email: mail@kalamunda.wa.gov.au</li>
                <li>• Address: 2 Railway Road, Kalamunda</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
          <div className="bg-gray-50 px-6 py-3 border-b">
            <h3 className="text-lg font-medium text-gray-900">Application Forms</h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {forms.map((form) => (
                <div key={form.code} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded-full">
                          {form.code}
                        </span>
                        <h4 className="font-medium text-gray-900">{form.name}</h4>
                      </div>
                      <p className="text-xs text-green-700 font-medium">Fee: {form.fee}</p>
                    </div>
                    <div className="flex gap-2">
                      <button className="text-blue-600 hover:text-blue-800 text-sm flex items-center gap-1">
                        <Download className="h-4 w-4" />
                        Download
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const FormCheckerModal = ({ application, onClose, systemReviewResults = null }) => {
    const [checklist, setChecklist] = useState(() => {
      if (systemReviewResults?.autoChecklist) {
        return systemReviewResults.autoChecklist;
      }
      return getFormChecklist(application.type).map(item => ({ ...item, checked: false, comments: '' }));
    });
    const [overallComments, setOverallComments] = useState(
      systemReviewResults?.systemRecommendations?.join('. ') || ''
    );
    const [recommendation, setRecommendation] = useState('');

    const requiredDocs = requiredDocuments[application.type] || [];
    const submittedDocs = application.documents || [];
    
    const toggleChecklistItem = (index) => {
      setChecklist(prev => prev.map((item, i) => 
        i === index ? { ...item, checked: !item.checked } : item
      ));
    };

    const updateComments = (index, comments) => {
      setChecklist(prev => prev.map((item, i) => 
        i === index ? { ...item, comments } : item
      ));
    };

    const requiredItemsChecked = checklist.filter(item => item.required).every(item => item.checked);
    const canApprove = requiredItemsChecked && recommendation;

    const submitReview = (decision) => {
      const reviewData = {
        checklist,
        overallComments,
        recommendation,
        decision,
        reviewDate: new Date().toISOString().split('T')[0],
        reviewer: 'Current Officer'
      };
      
      const newStatus = decision === 'approve' ? 'Approved' : 
                       decision === 'reject' ? 'Refused' : 'Requires Changes';
      
      setApplications(prev => prev.map(app => 
        app.id === application.id 
          ? { ...app, status: newStatus, review: reviewData }
          : app
      ));
      
      alert(`Application ${decision === 'approve' ? 'approved' : decision === 'reject' ? 'refused' : 'returned for changes'}`);
      onClose();
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-lg max-w-5xl w-full max-h-[90vh] overflow-y-auto">
          <div className="p-6 border-b border-gray-200">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Form Compliance Check - {application.id}
                </h2>
                {systemReviewResults && (
                  <p className="text-sm text-purple-600 mt-1 flex items-center gap-1">
                    <AlertTriangle className="h-4 w-4" />
                    Pre-populated with system analysis results
                  </p>
                )}
              </div>
              <button onClick={onClose} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>
          </div>
          
          <div className="p-6 space-y-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-2">Application Summary</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div><span className="font-medium">Type:</span> {application.type}</div>
                <div><span className="font-medium">Property:</span> {application.property}</div>
                <div><span className="font-medium">Applicant:</span> {application.applicant}</div>
                <div><span className="font-medium">Status:</span> 
                  <span className={`ml-2 px-2 py-1 rounded-full text-xs ${getStatusColor(application.status)}`}>
                    {application.status}
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Document Verification</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-700 mb-3">Required Documents</h4>
                  <div className="space-y-2">
                    {requiredDocs.map((doc, index) => {
                      const isSubmitted = submittedDocs.includes(doc);
                      return (
                        <div key={index} className="flex items-center gap-2">
                          <div className={`w-4 h-4 rounded-full ${isSubmitted ? 'bg-green-500' : 'bg-red-500'}`} />
                          <span className={`text-sm ${isSubmitted ? 'text-gray-900' : 'text-red-600'}`}>
                            {doc}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-700 mb-3">Submitted Documents</h4>
                  <div className="space-y-2">
                    {submittedDocs.map((doc, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span className="text-sm text-gray-900">{doc}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Compliance Checklist</h3>
              <div className="space-y-4">
                {checklist.map((item, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        checked={item.checked}
                        onChange={() => toggleChecklistItem(index)}
                        className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-medium ${item.checked ? 'text-green-700' : 'text-gray-900'}`}>
                            {item.item}
                          </span>
                          {item.required && (
                            <span className="text-red-500 text-xs">*Required</span>
                          )}
                        </div>
                        <textarea
                          placeholder="Add comments..."
                          className="mt-2 w-full px-3 py-2 border border-gray-300 rounded text-sm"
                          rows={2}
                          value={item.comments}
                          onChange={(e) => updateComments(index, e.target.value)}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Overall Assessment</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Officer Comments
                  </label>
                  <textarea
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                    rows={4}
                    value={overallComments}
                    onChange={(e) => setOverallComments(e.target.value)}
                    placeholder="Provide overall assessment comments..."
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Recommendation
                  </label>
                  <select
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                    value={recommendation}
                    onChange={(e) => setRecommendation(e.target.value)}
                  >
                    <option value="">Select recommendation</option>
                    <option value="approve">Recommend Approval</option>
                    <option value="approve_with_conditions">Approve with Conditions</option>
                    <option value="request_changes">Request Changes</option>
                    <option value="refuse">Recommend Refusal</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="flex justify-between items-center pt-4 border-t border-gray-200">
              <div className="text-sm text-gray-600">
                {requiredItemsChecked ? (
                  <span className="text-green-600 flex items-center gap-1">
                    <CheckCircle className="w-4 h-4" />
                    All required items checked
                  </span>
                ) : (
                  <span className="text-red-600 flex items-center gap-1">
                    <AlertTriangle className="w-4 h-4" />
                    Complete required items before proceeding
                  </span>
                )}
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={onClose}
                  className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={() => submitReview('changes')}
                  disabled={!recommendation}
                  className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Request Changes
                </button>
                <button
                  onClick={() => submitReview('reject')}
                  disabled={!recommendation}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Refuse
                </button>
                <button
                  onClick={() => submitReview('approve')}
                  disabled={!canApprove}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Approve
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const ApplicationDetail = ({ application, onClose }) => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-200">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold text-gray-900">
              Application Details - {application.id}
            </h2>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
              ✕
            </button>
          </div>
        </div>
        
        <div className="p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Application Information</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium text-gray-500">Type</label>
                  <p className="text-sm text-gray-900">{application.type}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Property</label>
                  <p className="text-sm text-gray-900">{application.property}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Applicant</label>
                  <p className="text-sm text-gray-900">{application.applicant}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Description</label>
                  <p className="text-sm text-gray-900">{application.description}</p>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Status & Timeline</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium text-gray-500">Current Status</label>
                  <p className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(application.status)}`}>
                    {application.status}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Submission Date</label>
                  <p className="text-sm text-gray-900">{application.submissionDate}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Target Date</label>
                  <p className="text-sm text-gray-900">{application.targetDate}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Assigned Officer</label>
                  <p className="text-sm text-gray-900">{application.assignedOfficer}</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
            <button 
              onClick={() => {
                setShowSystemReviewModal(true);
              }}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2"
            >
              <AlertTriangle className="h-4 w-4" />
              System Review
            </button>
            <button 
              onClick={() => {
                setShowFormCheckerModal(true);
              }}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
            >
              <CheckCircle className="h-4 w-4" />
              Check Forms
            </button>
            <button className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
              Update Status
            </button>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              Add Note
            </button>
          </div>
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
        alert('Please fill in all required fields');
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

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Building className="h-8 w-8 text-blue-600 mr-3" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">City of Kalamunda</h1>
                <p className="text-sm text-gray-600">Building Approval System</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">Building Act 2011 Compliant</span>
              <Users className="h-5 w-5 text-gray-400" />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'dashboard'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('applications')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'applications'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Applications
            </button>
            <button
              onClick={() => setActiveTab('review')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'review'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <CheckCircle className="h-4 w-4 inline mr-1" />
              Review Queue
            </button>
            <button
              onClick={() => setActiveTab('forms')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'forms'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <FileText className="h-4 w-4 inline mr-1" />
              Forms Reference
            </button>
            <button
              onClick={() => setActiveTab('calendar')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'calendar'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Calendar className="h-4 w-4 inline mr-1" />
              Council Calendar
            </button>
          </nav>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && <DashboardView />}
        {activeTab === 'applications' && <ApplicationsList />}
        {activeTab === 'review' && <ReviewQueue />}
        {activeTab === 'forms' && <FormsReference />}
        {activeTab === 'calendar' && (
          <div className="bg-white p-8 rounded-lg shadow-sm border text-center">
            <Calendar className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Council Meeting Calendar</h3>
            <p className="text-gray-600 mb-4">Next Council meeting: Fourth Monday of each month</p>
            <p className="text-sm text-gray-500">Applications requiring Council determination will be scheduled here</p>
          </div>
        )}
      </div>

      {selectedApplication && !showFormCheckerModal && !showSystemReviewModal && (
        <ApplicationDetail
          application={selectedApplication}
          onClose={() => setSelectedApplication(null)}
        />
      )}
      
      {showNewApplicationModal && (
        <NewApplicationModal onClose={() => setShowNewApplicationModal(false)} />
      )}
      
      {showSystemReviewModal && selectedApplication && (
        <SystemReviewModal
          application={selectedApplication}
          onClose={() => {
            setShowSystemReviewModal(false);
            setSelectedApplication(null);
          }}
          onApplyToFormChecker={(systemResults) => {
            setShowSystemReviewModal(false);
            setShowFormCheckerModal(true);
            // Pass system results to form checker
            setSelectedApplication(prev => ({ ...prev, systemReviewResults: systemResults }));
          }}
        />
      )}
      
      {showFormCheckerModal && selectedApplication && (
        <FormCheckerModal
          application={selectedApplication}
          onClose={() => {
            setShowFormCheckerModal(false);
            setSelectedApplication(null);
          }}
          systemReviewResults={selectedApplication.systemReviewResults}
        />
      )}
    </div>
  );
};

export default BuildingApprovalSystem;
