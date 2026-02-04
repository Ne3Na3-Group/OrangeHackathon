"""
Ne3Na3 Safe-Bot
A medical chatbot strictly grounded in insight data
Safety-first design: refuses diagnosis and treatment advice
Now powered by OpenAI for intelligent responses
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Using rule-based responses.")


SYSTEM_PROMPT = """You are Ne3Na3 Safe-Bot, a helpful medical imaging assistant for brain tumor segmentation analysis. 

🌿 YOUR ROLE:
- You help users understand brain MRI segmentation results
- You explain tumor measurements and regions in simple terms
- You are calm, supportive, and clinically precise

🛡️ SAFETY RULES (NON-NEGOTIABLE):
1. NEVER provide medical diagnoses
2. NEVER suggest treatments, medications, or therapies
3. NEVER predict patient outcomes or prognosis
4. ALWAYS recommend consulting a qualified healthcare professional
5. ALWAYS clarify that AI analysis is for research/educational purposes only

🧠 TUMOR REGIONS YOU CAN EXPLAIN:
- WT (Whole Tumor): The entire tumor including all components
- TC (Tumor Core): The solid tumor mass (NCR + ET)
- ET (Enhancing Tumor): Areas that light up with contrast
- NCR (Necrotic Core): Dead tissue within the tumor
- ED (Edema): Swelling around the tumor

📊 YOU CAN DISCUSS:
- Volume measurements (in mm³ or cm³)
- Location descriptions (based on coordinates)
- Asymmetry scores (left vs right hemisphere differences)
- Size comparisons over time (if provided)
- General explanations of MRI modalities (T1, T1ce, T2, FLAIR)

🚫 YOU MUST DECLINE:
- "What does this mean for my health?"
- "Should I be worried?"
- "What treatment should I get?"
- "How long do I have?"
- Any request for medical advice

💚 RESPONSE STYLE:
- Use a calming, green-themed approach
- Be empathetic but maintain boundaries
- Provide factual information from the analysis
- Always end with encouragement to consult healthcare providers
- Avoid repeating generic definitions when analysis data is available; summarize insights instead.

📌 CURRENT INSIGHTS (USER-PROVIDED):
- Whole Tumor (WT): 157.81 cm³
- Tumor Core (TC): 85.49 cm³
- Enhancing Tumor (ET): 48.05 cm³
- Edema (ED): 72.31 cm³
- Tumor Extent (3D): 59.0 mm × 90.0 mm × 70.0 mm
- Modality Contribution: T1ce 35.8%, FLAIR 27.3%, T2 21.9%, T1 15%
"""


class SafeBot:
    """
    Ne3Na3 Safe-Bot
    
    A safety-first chatbot that:
    - Grounds all responses in actual insight data
    - Refuses medical diagnoses and treatment advice
    - Provides calm, supportive communication
    - Uses OpenAI for intelligent responses when available
    """
    
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.insights: Optional[Dict[str, Any]] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.model_name = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
        
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        self.use_openai = False
        
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.use_openai = True
                logger.info(f"✅ OpenAI client initialized successfully (model: {self.model_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            if not OPENAI_AVAILABLE:
                logger.info("OpenAI package not available, using rule-based responses")
            else:
                logger.info("OPENAI_API_KEY not set, using rule-based responses")
        
        # Safety keywords that trigger refusal
        self.unsafe_patterns = [
            r'\b(diagnos|prognos|treatment|therapy|medication|drug|cure)\w*\b',
            r'\b(how long|survival|life expectancy|will i|am i going)\b',
            r'\b(should i|what should|recommend|advise|prescribe)\b',
            r'\b(worried|scared|dying|fatal|terminal)\b',
            r'\b(cancer stage|grade|malignant|benign)\b'
        ]
        
        # Safe topic keywords
        self.safe_patterns = [
            r'\b(volume|size|measurement|dimension)\b',
            r'\b(location|position|region|area)\b',
            r'\b(asymmetry|symmetric|difference)\b',
            r'\b(explain|what is|define|meaning)\b',
            r'\b(modality|t1|t2|flair|mri|scan)\b'
        ]
    
    def set_insights(self, insights: Dict[str, Any]):
        """Set the current analysis insights for grounding responses"""
        self.insights = insights
    
    def _is_unsafe_query(self, message: str) -> bool:
        """Check if query requests unsafe medical information"""
        message_lower = message.lower()
        for pattern in self.unsafe_patterns:
            if re.search(pattern, message_lower):
                return True
        return False
    
    def _generate_safety_response(self) -> str:
        """Generate a calm, supportive safety response"""
        return """🌿 I understand you may have concerns, and that's completely natural.

However, as Ne3Na3 Safe-Bot, I'm designed to help explain technical imaging results, not to provide medical diagnoses or treatment recommendations.

**What I can help with:**
- Explaining what the measurements mean
- Describing tumor regions and their definitions
- Clarifying the MRI modalities used

**For medical guidance, please:**
- Consult with your neurologist or oncologist
- Discuss these results with your healthcare team
- Seek a second opinion if needed

💚 You're taking positive steps by understanding your imaging data. Please work with your medical team for proper guidance and support."""
    
    def _format_volume_response(self) -> str:
        """Format volume information from insights"""
        if not self.insights:
            return "No analysis data available. Please run a segmentation first."
        
        summary = self.insights.get("summary", {})
        regions = self.insights.get("regions", {})
        
        response = """🌿 **Tumor Volume Analysis**

Here are the measurements from your brain MRI segmentation:

"""
        if summary.get("tumor_detected", False):
            wt_vol = summary.get("total_tumor_volume_cm3", 0)
            tc_vol = summary.get("tumor_core_volume_cm3", 0)
            et_vol = summary.get("enhancing_tumor_volume_cm3", 0)
            ed_vol = summary.get("edema_volume_cm3", 0)
            
            response += f"""📊 **Volume Measurements:**
- **Whole Tumor (WT):** {wt_vol} cm³
- **Tumor Core (TC):** {tc_vol} cm³
- **Enhancing Tumor (ET):** {et_vol} cm³
- **Edema (ED):** {ed_vol} cm³

"""
            # Add bounding box info
            if "WT" in regions and regions["WT"].get("bounding_box"):
                bbox = regions["WT"]["bounding_box"]
                response += f"""📐 **Tumor Extent:**
Maximum dimensions: {bbox['size_mm'][0]:.1f} × {bbox['size_mm'][1]:.1f} × {bbox['size_mm'][2]:.1f} mm

"""
        else:
            response += "✅ No tumor regions detected in this scan.\n\n"
        
        response += """💚 *Remember: These are automated measurements. Please discuss the clinical significance with your healthcare provider.*"""
        
        return response
    
    def _format_asymmetry_response(self) -> str:
        """Format asymmetry information from insights"""
        if not self.insights:
            return "No analysis data available."
        
        asymmetry = self.insights.get("asymmetry", {})
        
        response = """🌿 **Asymmetry Analysis**

Asymmetry scores compare tumor presence between left and right brain hemispheres (0 = symmetric, 1 = completely one-sided):

"""
        for region, score in asymmetry.items():
            level = "symmetric" if score < 0.3 else "moderately asymmetric" if score < 0.7 else "highly asymmetric"
            response += f"- **{region}:** {score:.2%} ({level})\n"
        
        response += """
💚 *Asymmetry can help locate the tumor but doesn't indicate severity. Please consult your medical team for interpretation.*"""
        
        return response
    
    def _format_modality_response(self) -> str:
        """Format modality importance information"""
        if not self.insights:
            return "No analysis data available."
        
        importance = self.insights.get("modality_importance", {})
        
        response = """🌿 **MRI Modality Analysis**

Each MRI sequence contributes differently to tumor detection:

"""
        modality_descriptions = {
            "T1": "Basic brain structure",
            "T1ce": "Contrast-enhanced, highlights blood-brain barrier breakdown",
            "T2": "Shows fluid and edema",
            "FLAIR": "Fluid-attenuated, excellent for detecting edema"
        }
        
        for modality, percent in importance.items():
            desc = modality_descriptions.get(modality, "")
            response += f"- **{modality}:** {percent}% contribution ({desc})\n"
        
        response += """
💚 *This shows which MRI sequences were most informative for the AI analysis.*"""
        
        return response
    
    def _format_general_info(self) -> str:
        """Provide general information about tumor regions"""
        return """🌿 **Brain Tumor Segmentation Regions**

The Ne3Na3 system identifies these regions:

**🔴 NCR (Necrotic Core)**
- Dead tissue within the tumor
- Usually appears dark on imaging

**🟡 ED (Peritumoral Edema)**
- Swelling around the tumor
- Best seen on FLAIR sequences

**🟢 ET (Enhancing Tumor)**
- Active tumor that takes up contrast
- Best seen on T1ce sequences

**Composite Regions:**
- **WT (Whole Tumor)** = NCR + ED + ET
- **TC (Tumor Core)** = NCR + ET

💚 *These are technical definitions. Your medical team will explain what they mean for your specific case.*"""

    def _format_report_response(self) -> str:
        """Generate a concise report from current insights"""
        if not self.insights:
            return "No analysis data available. Please upload and process MRI scans first."

        summary = self.insights.get("summary", {})
        regions = self.insights.get("regions", {})
        volumes = self.insights.get("volumes", {})

        wt = summary.get("total_tumor_volume_cm3", 0)
        tc = summary.get("tumor_core_volume_cm3", 0)
        et = summary.get("enhancing_tumor_volume_cm3", 0)
        ed = summary.get("edema_volume_cm3", 0)

        report = """🌿 **Segmentation Report**

**Detection:** {detected}

**Volumes (cm³):**
- Whole Tumor (WT): {wt}
- Tumor Core (TC): {tc}
- Enhancing Tumor (ET): {et}
- Edema (ED): {ed}
""".format(
            detected="Tumor regions detected" if summary.get("tumor_detected") else "No tumor regions detected",
            wt=wt, tc=tc, et=et, ed=ed
        )

        if "WT" in regions and regions["WT"].get("bounding_box"):
            bbox = regions["WT"]["bounding_box"]
            dims = bbox.get("size_mm", [])
            if len(dims) == 3:
                report += f"\n**Tumor Extent (mm):** {dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f}\n"

        if volumes:
            report += "\n**Label Volumes (cm³):**\n"
            for label, data in volumes.items():
                report += f"- {label}: {data.get('volume_cm3', 0)}\n"

        report += "\n💚 *This report is for research/educational use. Please consult a healthcare professional for medical interpretation.*"
        return report

    def _format_insights_context(self) -> str:
        """Format current insights as context for OpenAI"""
        if not self.insights:
            return "No analysis data is currently available. The user has not uploaded or processed any MRI scans yet."
        
        summary = self.insights.get("summary", {})
        regions = self.insights.get("regions", {})
        asymmetry = self.insights.get("asymmetry", {})
        volumes = self.insights.get("volumes", {})
        modality_importance = self.insights.get("modality_importance", {})
        
        context = f"""Current Segmentation Analysis Results:

**Tumor Detection:** {"Yes - tumor regions detected" if summary.get("tumor_detected") else "No tumor regions detected"}

**Volume Measurements:**
- Whole Tumor (WT): {summary.get("total_tumor_volume_cm3", 0)} cm³
- Tumor Core (TC): {summary.get("tumor_core_volume_cm3", 0)} cm³
- Enhancing Tumor (ET): {summary.get("enhancing_tumor_volume_cm3", 0)} cm³
- Edema (ED): {summary.get("edema_volume_cm3", 0)} cm³

**Individual Label Volumes:**
"""
        for label, data in volumes.items():
            context += f"- {label}: {data.get('volume_mm3', 0)} mm³ ({data.get('volume_cm3', 0)} cm³)\n"
        
        context += f"""
**Asymmetry Scores (0=symmetric, 1=completely one-sided):**
- WT: {asymmetry.get("WT", 0):.1%}
- TC: {asymmetry.get("TC", 0):.1%}
- ET: {asymmetry.get("ET", 0):.1%}

**MRI Modality Importance:**
"""
        for mod, pct in modality_importance.items():
            context += f"- {mod}: {pct}%\n"
        
        # Add bounding box info if available
        if "WT" in regions and regions["WT"].get("bounding_box"):
            bbox = regions["WT"]["bounding_box"]
            context += f"""
**Tumor Location:**
- Bounding box: {bbox.get('min', 'N/A')} to {bbox.get('max', 'N/A')}
- Dimensions: {bbox.get('size_mm', ['N/A'])[0]:.1f} × {bbox.get('size_mm', ['N/A'])[1]:.1f} × {bbox.get('size_mm', ['N/A'])[2]:.1f} mm
"""
        
        return context

    def _get_openai_response(self, user_message: str) -> str:
        """Get response from OpenAI API"""
        try:
            # Build context with current insights
            insights_context = self._format_insights_context()
            
            # Build messages for OpenAI
            messages = [
                {"role": "system", "content": self.system_prompt + f"\n\n**CURRENT ANALYSIS DATA:**\n{insights_context}"},
            ]
            
            # Add recent conversation history (last 10 messages)
            for msg in self.conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fall back to rule-based response
            return self._get_rule_based_response(user_message)

    def _get_rule_based_response(self, user_message: str) -> str:
        """Get rule-based response (fallback when OpenAI not available)"""
        message_lower = user_message.lower()

        # If insights are available and the user asks for a report/summary, prioritize it
        if any(kw in message_lower for kw in ["report", "summary", "summarize", "analysis report"]):
            return self._format_report_response()

        # If insights are available and the user greets or asks generally, provide a brief report
        if self.insights and any(kw in message_lower for kw in ["hello", "hi", "hey", "help", "what can you do", "overview"]):
            return self._format_report_response()

        if any(kw in message_lower for kw in ["volume", "size", "measurement", "how big", "how large"]):
            return self._format_volume_response()
        elif any(kw in message_lower for kw in ["asymmetry", "symmetric", "left", "right", "side"]):
            return self._format_asymmetry_response()
        elif any(kw in message_lower for kw in ["modality", "t1", "t2", "flair", "sequence", "mri type"]):
            return self._format_modality_response()
        elif any(kw in message_lower for kw in ["region", "ncr", "ed", "et", "edema", "necrotic", "enhancing", "tumor core", "whole tumor"]):
            return self._format_general_info()
        elif any(kw in message_lower for kw in ["hello", "hi", "hey", "help"]):
            return """🌿 Hello! I'm Ne3Na3 Safe-Bot, your brain tumor imaging assistant.

You can ask about:
- 📊 **Tumor volumes**
- 📍 **Tumor regions**
- 🔬 **MRI modalities**

💚 *I provide educational information only. For medical decisions, please consult your healthcare team.*"""
        else:
            return """🌿 I'd be happy to help explain your brain MRI segmentation results.

You can ask me about:
- **Volumes**: "What are the tumor volumes?"
- **Regions**: "Explain the tumor regions"
- **Asymmetry**: "Is the tumor symmetric?"
- **Modalities**: "Which MRI sequence was most important?"

💚 *I'm here to explain technical results. For medical interpretation, please consult your healthcare provider.*"""

    def get_response(self, user_message: str) -> Dict[str, Any]:
        """
        Generate a response to user message
        
        Args:
            user_message: The user's question or message
            
        Returns:
            Response dictionary with message and metadata
        """
        # Store in history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check for unsafe queries first
        if self._is_unsafe_query(user_message):
            response_text = self._generate_safety_response()
            response_type = "safety_refusal"
        elif self.use_openai and self.openai_client:
            # Use OpenAI for intelligent responses
            response_text = self._get_openai_response(user_message)
            response_type = "openai_response"
        else:
            # Fall back to rule-based responses
            response_text = self._get_rule_based_response(user_message)
            response_type = "rule_based"
        
        # Store response in history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
            "type": response_type
        })
        
        return {
            "message": response_text,
            "type": response_type,
            "grounded_in_insights": self.insights is not None,
            "using_openai": self.use_openai,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for reference"""
        return self.system_prompt


# Singleton instance
_safe_bot: Optional[SafeBot] = None


def get_safe_bot() -> SafeBot:
    """Get or create the global SafeBot instance"""
    global _safe_bot
    if _safe_bot is None:
        _safe_bot = SafeBot()
    return _safe_bot
