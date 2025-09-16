#%%
#!/usr/bin/env python3
"""
AI-Powered Support Ticket Analyzer
Uses Claude's reasoning to categorize, prioritize, and route tickets intelligently
"""

import chromadb
import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

@dataclass
class TicketAnalysis:
    """Result of ticket analysis"""
    category: str
    subcategory: str
    priority: str
    assigned_team: str
    technical_complexity: str
    required_expertise: List[str]
    escalation_required: bool
    reasoning: str
    confidence_score: float
    similar_tickets: List[Dict]
    immediate_actions: List[str]


class AITicketAnalyzer:
    """
    AI-powered ticket analyzer using Claude's reasoning capabilities
    """
    
    def __init__(self, db_path: str = "./topdesk_vectordb"):
        """Initialize with existing ChromaDB"""
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Check available collections
        available_collections = [col.name for col in self.client.list_collections()]
        print(f"Available collections: {available_collections}")
        
        # Try to get the incidents collection (check different possible names)
        collection_names = ["incidents", "topdesk_incidents", "topdesk-incidents"]
        self.collection = None
        
        for name in collection_names:
            try:
                self.collection = self.client.get_collection(name)
                print(f"✓ Connected to collection: {name}")
                break
            except:
                continue
        
        if not self.collection:
            raise ValueError(f"No incidents collection found. Available: {available_collections}")
    
    def get_similar_tickets(self, query: str, n_results: int = 8) -> List[Dict]:
        """Get similar tickets from ChromaDB with rich context"""
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        similar_tickets = []
        for doc, meta, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            similarity_score = 1 - distance
            
            # Only include tickets with reasonable similarity
            if similarity_score > 0.3:
                similar_tickets.append({
                    'similarity': similarity_score,
                    'category': meta.get('category', ''),
                    'subcategory': meta.get('subcategory', ''),
                    'priority': meta.get('priority', ''),
                    'brief_description': meta.get('brief_description', ''),
                    'department': meta.get('caller_department', ''),
                    'software': meta.get('software_required', ''),
                    'operator_group': meta.get('operator_group', ''),
                    'status': meta.get('status', ''),
                    'urgency': meta.get('urgency', ''),
                    'impact': meta.get('impact', ''),
                    'call_date': meta.get('call_date', ''),
                    'document_content': doc[:300] + "..." if len(doc) > 300 else doc
                })
        
        return similar_tickets
    
    def create_system_prompt(self) -> str:
        """Create comprehensive system prompt for Claude"""
        
        return """You are an expert IT Support Analyst with 15+ years of experience in enterprise technical support. Your expertise includes:

- Incident categorization and priority assessment
- Technical complexity evaluation 
- Team routing and workload distribution
- Escalation procedures and SLA management
- Risk assessment and business impact analysis

CORE RESPONSIBILITIES:
1. Analyze support tickets using historical patterns and technical knowledge
2. Determine appropriate categorization based on issue type and technical requirements
3. Assess priority considering business impact, user count, and urgency
4. Route tickets to teams based on technical expertise and workload capacity
5. Identify escalation needs and potential risks

AVAILABLE SUPPORT TEAMS & EXPERTISE:
- Level 1 Support: Basic troubleshooting, password resets, general user guidance, simple software issues
- Applications Support: Software installation/configuration, desktop applications, user training, RStudio, Python, R, Office suite
- Infrastructure Team: Network issues, server problems, storage systems, backup/recovery, performance issues, hardware
- HPC Team: High-performance computing, cluster management, job scheduling (SLURM/PBS), scientific computing, GPU workloads
- Security Team: Security incidents, access control, compliance, suspicious activity, data breaches
- Data Services: Database issues, data migration, ETL processes, data governance, BI tools

PRIORITY GUIDELINES:
- P1 (Critical): System outages, security breaches, data loss, complete service failures affecting multiple users/departments
- P2 (High): Significant functionality loss, performance issues affecting productivity, partial outages, approaching deadlines
- P3 (Medium): Moderate issues affecting individual users or small teams, software bugs with workarounds
- P4 (Low): Minor issues, enhancement requests, training requests, documentation updates

COMPLEXITY ASSESSMENT:
- Critical: Multi-system failures, security incidents, data integrity issues requiring senior expertise
- High: Complex configurations, system integrations, performance optimization, specialized technical knowledge needed
- Medium: Standard software issues, configuration changes, troubleshooting requiring technical skills
- Low: Basic user issues, simple configurations, routine maintenance tasks

Always base your analysis on:
1. Historical similar incidents and their outcomes
2. Technical complexity and required expertise
3. Business impact and user count affected
4. Urgency and timeline constraints
5. Available team capacity and expertise

Provide thorough reasoning for all decisions and maintain consistency with historical patterns while adapting to unique circumstances."""

    def create_analysis_prompt(self, query: str, similar_tickets: List[Dict]) -> str:
        """Create comprehensive analysis prompt"""
        
        # Build rich context from similar tickets
        context_sections = []
        
        if similar_tickets:
            context_sections.append("=== SIMILAR HISTORICAL TICKETS ===")
            
            for i, ticket in enumerate(similar_tickets, 1):
                context_sections.append(f"""
TICKET {i} (Similarity: {ticket['similarity']:.1%})
Category: {ticket['category']} > {ticket['subcategory']}
Priority: {ticket['priority']} | Urgency: {ticket['urgency']} | Impact: {ticket['impact']}
Team: {ticket['operator_group']} | Status: {ticket['status']}
Department: {ticket['department']} | Software: {ticket['software']}
Date: {ticket['call_date']}
Description: {ticket['brief_description']}
Context: {ticket['document_content']}
---""")
        
        context = '\n'.join(context_sections)
        
        prompt = f"""{context}

=== NEW SUPPORT TICKET TO ANALYZE ===
{query}

=== ANALYSIS REQUEST ===
Based on your expertise and the historical patterns above, provide a comprehensive analysis of this support ticket. Consider:

1. What type of issue is this and how does it compare to similar historical cases?
2. What is the appropriate category and subcategory based on the technical nature?
3. What priority should this receive considering business impact, urgency, and affected users?
4. Which team has the right expertise and capacity to handle this effectively?
5. What is the technical complexity and what specific skills are needed?
6. Does this require immediate escalation or special attention?
7. What immediate actions should be taken to begin resolution?

Respond with your analysis in this exact JSON structure:

{{
    "category": "primary category based on issue type",
    "subcategory": "specific subcategory for precise classification",
    "priority": "P1/P2/P3/P4 with clear justification",
    "assigned_team": "team with best expertise match",
    "technical_complexity": "critical/high/medium/low based on technical requirements",
    "required_expertise": ["specific", "technical", "skills", "needed"],
    "escalation_required": true/false,
    "reasoning": "comprehensive explanation of your analysis, referencing similar tickets and technical considerations",
    "confidence_score": 0.85,
    "immediate_actions": ["first action", "second action", "third action"],
    "estimated_resolution_time": "realistic timeframe based on complexity and team capacity",
    "business_impact": "assessment of impact on operations and users",
    "risk_factors": ["potential risks or complications to watch for"]
}}

Use your expertise to make intelligent decisions. Reference the similar tickets where relevant, but adapt to the unique aspects of this new issue. Be thorough in your reasoning and confident in your recommendations."""

        return prompt
    
    def call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic Claude with system and user prompts"""
        
        try:
            import anthropic
            
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent analysis
                system=system_prompt,
                messages=[
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except ImportError:
            return '{"error": "Install anthropic package: pip install anthropic"}'
        except Exception as e:
            return f'{{"error": "Anthropic API error: {str(e)}"}}'
    
    def analyze_ticket(self, query: str) -> TicketAnalysis:
        """
        Complete AI-powered ticket analysis
        
        Args:
            query: Support ticket description
            
        Returns:
            TicketAnalysis with intelligent recommendations
        """
        
        print(f"🔍 Analyzing: {query[:80]}...")
        
        # Step 1: Get similar tickets from vector database
        similar_tickets = self.get_similar_tickets(query)
        print(f"📊 Found {len(similar_tickets)} similar tickets for context")
        
        # Step 2: Create system and user prompts
        system_prompt = self.create_system_prompt()
        user_prompt = self.create_analysis_prompt(query, similar_tickets)
        
        # Step 3: Get Claude's intelligent analysis
        print("🤖 Getting Claude's expert analysis...")
        claude_response = self.call_claude(system_prompt, user_prompt)
        
        # Step 4: Parse and structure results
        try:
            analysis_data = json.loads(claude_response)
            
            return TicketAnalysis(
                category=analysis_data.get("category", "Unknown"),
                subcategory=analysis_data.get("subcategory", "Unknown"),
                priority=analysis_data.get("priority", "P3"),
                assigned_team=analysis_data.get("assigned_team", "Level 1 Support"),
                technical_complexity=analysis_data.get("technical_complexity", "medium"),
                required_expertise=analysis_data.get("required_expertise", []),
                escalation_required=analysis_data.get("escalation_required", False),
                reasoning=analysis_data.get("reasoning", "AI analysis completed"),
                confidence_score=analysis_data.get("confidence_score", 0.7),
                similar_tickets=similar_tickets[:3],
                immediate_actions=analysis_data.get("immediate_actions", [])
            )
            
        except json.JSONDecodeError:
            print("⚠️  JSON parsing failed, extracting key information...")
            
            # Fallback: extract information from raw response
            return TicketAnalysis(
                category="Analysis Error",
                subcategory="JSON Parse Failed",
                priority="P3",
                assigned_team="Level 1 Support",
                technical_complexity="medium",
                required_expertise=["manual review"],
                escalation_required=True,
                reasoning=f"Could not parse Claude response. Raw response: {claude_response[:200]}...",
                confidence_score=0.3,
                similar_tickets=similar_tickets[:3],
                immediate_actions=["Manual review required", "Check response format"]
            )
    
    def batch_analyze(self, tickets: List[str]) -> List[TicketAnalysis]:
        """Analyze multiple tickets"""
        
        results = []
        for i, ticket in enumerate(tickets, 1):
            print(f"\n{'='*60}")
            print(f"PROCESSING TICKET {i}/{len(tickets)}")
            print(f"{'='*60}")
            
            analysis = self.analyze_ticket(ticket)
            results.append(analysis)
            
            print(f"✅ Completed: {analysis.category} | {analysis.priority} | {analysis.assigned_team}")
        
        return results


def main():
    """Example usage with AI-powered analysis"""
    
    print("=== AI-Powered Support Ticket Analyzer ===")
    print("Using Claude's intelligent reasoning for ticket analysis\n")
    
    # Initialize analyzer
    try:
        analyzer = AITicketAnalyzer()
        print("✅ Connected to ChromaDB successfully")
    except Exception as e:
        print(f"❌ Error: Could not connect to ChromaDB: {e}")
        print("Make sure your ChromaDB is at ./topdesk_vectordb")
        return
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return
    else:
        print("✅ Anthropic API key found")
    
    # Example support tickets with varying complexity
    test_tickets = [
        """RStudio Server is completely down across the entire university. Multiple research groups in Psychology, Biology, and Statistics departments cannot access their R sessions. We have critical research deadlines this week and students cannot complete their coursework. The service has been unavailable for 2 hours.""",
        
        """Getting intermittent permission denied errors when accessing the shared research drive via VPN from home. Started happening after yesterday's network maintenance. About 15 users in the Engineering department are affected, but they can still work with local files. Some users report it works sometimes but fails other times.""",
        
        """HPC cluster job keeps failing with CUDA out-of-memory errors on the GPU nodes. The same job configuration worked perfectly last week with identical datasets. Need help optimizing memory allocation for deep learning training. This is blocking our Nature paper submission.""",
        
        """New postdoc researcher needs Python packages installed on their workstation - specifically TensorFlow, PyTorch, and Jupyter. They're getting administrator privilege errors when trying to install via pip. Also need guidance on connecting to the department's shared computing resources.""",
        
        """Security alert: Detected multiple suspicious login attempts on 20+ user accounts overnight from IP addresses in Eastern Europe. No successful logins yet, but seeing unusual patterns. Need immediate investigation and potentially disable affected accounts as precaution."""
    ]
    
    print(f"\n🚀 Analyzing {len(test_tickets)} tickets with Claude's AI reasoning...\n")
    
    for i, ticket in enumerate(test_tickets, 1):
        print(f"{'='*80}")
        print(f"TICKET {i}")
        print(f"{'='*80}")
        print(f"Description: {ticket}\n")
        
        # Analyze ticket with AI
        analysis = analyzer.analyze_ticket(ticket)
        
        # Display comprehensive results
        print(f"🎯 INTELLIGENT CATEGORIZATION:")
        print(f"   Category: {analysis.category}")
        print(f"   Subcategory: {analysis.subcategory}")
        
        print(f"\n⚡ AI PRIORITY ASSESSMENT:")
        print(f"   Priority: {analysis.priority}")
        print(f"   Technical Complexity: {analysis.technical_complexity}")
        print(f"   Escalation Required: {'🚨 YES' if analysis.escalation_required else '✅ NO'}")
        
        print(f"\n👥 EXPERT TEAM ROUTING:")
        print(f"   Assigned Team: {analysis.assigned_team}")
        print(f"   Required Expertise: {', '.join(analysis.required_expertise)}")
        print(f"   AI Confidence: {analysis.confidence_score:.1%}")
        
        print(f"\n🧠 CLAUDE'S REASONING:")
        print(f"   {analysis.reasoning}")
        
        print(f"\n⚡ IMMEDIATE ACTIONS:")
        for j, action in enumerate(analysis.immediate_actions, 1):
            print(f"   {j}. {action}")
        
        print(f"\n📊 SIMILAR HISTORICAL CASES:")
        for j, similar in enumerate(analysis.similar_tickets, 1):
            print(f"   {j}. {similar['similarity']:.1%} similarity - {similar['brief_description'][:50]}...")
            print(f"      Previous team: {similar['operator_group']}, Priority: {similar['priority']}")
        
        print("\n")
    
    print("=" * 80)
    print("🎉 AI ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nThe AI analyzer provides:")
    print("✅ Intelligent categorization using Claude's reasoning")
    print("✅ Context-aware priority assessment") 
    print("✅ Expert team routing based on technical requirements")
    print("✅ Comprehensive analysis with actionable recommendations")
    print("✅ Historical pattern recognition and learning")


if __name__ == "__main__":
    main()
# %%
