"""
Supabase client for authentication and subscription management.
Uses individual packages (supabase-auth, postgrest) instead of full supabase package.
"""

import logging
import webbrowser
from datetime import datetime
from typing import Optional, Dict, Any

# Try to import the individual packages
try:
    from gotrue import SyncGoTrueClient
    from postgrest import SyncPostgrestClient
    SUPABASE_AVAILABLE = True
except ImportError:
    try:
        # Alternative import paths
        from supabase_auth import SyncGoTrueClient
        from postgrest import SyncPostgrestClient
        SUPABASE_AVAILABLE = True
    except ImportError:
        SUPABASE_AVAILABLE = False
        SyncGoTrueClient = None
        SyncPostgrestClient = None

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = "https://gsvccxhdgcshiwgjvgfi.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdzdmNjeGhkZ2NzaGl3Z2p2Z2ZpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjczOTY2NTIsImV4cCI6MjA4Mjk3MjY1Mn0.Sbb6YJjlQ_ig2LCcs9zz_Be1kU-iIHBx4Vu4nzCPyTM"

# Stripe configuration - Pricing Plans
STRIPE_PRICE_ID_STARTER = "price_1SeEv5BATYQXewwiQ5XO32PD"  # Starter plan
STRIPE_PRICE_ID_ULTRA = "price_1SuJOxBATYQXewwiuqsqAcMJ"  # Ultra plan $49/month
STRIPE_PRICE_ID = STRIPE_PRICE_ID_STARTER  # Default for checkout

# Redirect URLs for authentication flows
SITE_URL = "https://filect.io"
REDIRECT_URL_SIGNUP = f"{SITE_URL}/signup-success"
REDIRECT_URL_PASSWORD_RESET = f"{SITE_URL}/secret-reset-password"
REDIRECT_URL_PAYMENT_SUCCESS = f"{SITE_URL}/payment-success"

# Index limits per plan (images, videos, audio only - text files are unlimited)
INDEX_LIMIT_STARTER = 1000   # 1000 media files per month for starter
INDEX_LIMIT_ULTRA = 5000     # 5000 media files per month for ultra


class SupabaseAuth:
    """Handles Supabase authentication and subscription management."""
    
    def __init__(self):
        self._auth_client = None
        self._db_client = None
        self._user: Optional[Dict[str, Any]] = None
        self._session: Optional[Dict[str, Any]] = None
        self._subscription: Optional[Dict[str, Any]] = None
        self._access_token: Optional[str] = None
        
        if SUPABASE_AVAILABLE:
            try:
                # Initialize GoTrue client for authentication
                self._auth_client = SyncGoTrueClient(
                    url=f"{SUPABASE_URL}/auth/v1",
                    headers={"apikey": SUPABASE_ANON_KEY}
                )
                logger.info("Supabase auth client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase auth client: {e}")
        else:
            logger.warning("Supabase packages not installed. Run: pip install postgrest supabase-auth httpx")
    
    def _get_db_client(self) -> Optional[SyncPostgrestClient]:
        """Get a PostgREST client with current auth token."""
        if not SUPABASE_AVAILABLE:
            return None
        
        headers = {"apikey": SUPABASE_ANON_KEY}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        
        return SyncPostgrestClient(
            base_url=f"{SUPABASE_URL}/rest/v1",
            headers=headers
        )
    
    @property
    def is_available(self) -> bool:
        """Check if Supabase client is available."""
        return self._auth_client is not None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        return self._user is not None and self._session is not None
    
    @property
    def current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user."""
        return self._user
    
    @property
    def user_email(self) -> Optional[str]:
        """Get current user's email."""
        if self._user:
            return self._user.get('email')
        return None
    
    def _extract_user_dict(self, user_obj) -> Dict[str, Any]:
        """Extract user data from response object."""
        if hasattr(user_obj, 'model_dump'):
            return user_obj.model_dump()
        elif hasattr(user_obj, '__dict__'):
            return user_obj.__dict__
        elif isinstance(user_obj, dict):
            return user_obj
        else:
            return {'id': str(user_obj)}
    
    def _extract_session_dict(self, session_obj) -> Dict[str, Any]:
        """Extract session data from response object."""
        if session_obj is None:
            return {}
        if hasattr(session_obj, 'model_dump'):
            return session_obj.model_dump()
        elif hasattr(session_obj, '__dict__'):
            return session_obj.__dict__
        elif isinstance(session_obj, dict):
            return session_obj
        else:
            return {}
    
    def sign_up(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign up a new user.
        
        Returns:
            dict with 'success' bool and 'error' or 'user' keys
        """
        if not self._auth_client:
            return {'success': False, 'error': 'Supabase not available'}
        
        try:
            response = self._auth_client.sign_up({
                'email': email,
                'password': password,
                'options': {
                    'email_redirect_to': REDIRECT_URL_SIGNUP
                }
            })
            
            if response.user:
                self._user = self._extract_user_dict(response.user)
                if response.session:
                    self._session = self._extract_session_dict(response.session)
                    self._access_token = self._session.get('access_token')
                else:
                    self._session = None
                logger.info(f"User signed up: {email}")
                return {'success': True, 'user': self._user, 'needs_confirmation': response.session is None}
            else:
                return {'success': False, 'error': 'Sign up failed'}
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sign up error: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in an existing user.
        
        Returns:
            dict with 'success' bool and 'error' or 'user' keys
        """
        if not self._auth_client:
            return {'success': False, 'error': 'Supabase not available'}
        
        try:
            response = self._auth_client.sign_in_with_password({
                'email': email,
                'password': password
            })
            
            if response.user and response.session:
                self._user = self._extract_user_dict(response.user)
                self._session = self._extract_session_dict(response.session)
                self._access_token = self._session.get('access_token')
                logger.info(f"User signed in: {email}")
                return {'success': True, 'user': self._user}
            else:
                return {'success': False, 'error': 'Sign in failed'}
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sign in error: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def sign_out(self) -> Dict[str, Any]:
        """Sign out current user."""
        if not self._auth_client:
            return {'success': False, 'error': 'Supabase not available'}
        
        try:
            self._auth_client.sign_out()
            self._user = None
            self._session = None
            self._subscription = None
            self._access_token = None
            logger.info("User signed out")
            return {'success': True}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sign out error: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def reset_password(self, email: str) -> Dict[str, Any]:
        """
        Send password reset email to user.
        
        Args:
            email: User's email address
            
        Returns:
            dict with 'success' bool and optional 'error' message
        """
        if not self._auth_client:
            return {'success': False, 'error': 'Supabase not available'}
        
        try:
            self._auth_client.reset_password_for_email(
                email,
                options={
                    'redirect_to': REDIRECT_URL_PASSWORD_RESET
                }
            )
            logger.info(f"Password reset email sent to: {email}")
            return {'success': True}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Password reset error: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def restore_session(self, access_token: str, refresh_token: str) -> Dict[str, Any]:
        """
        Restore a session from stored tokens.
        
        Returns:
            dict with 'success' bool
        """
        if not self._auth_client:
            return {'success': False, 'error': 'Supabase not available'}
        
        try:
            response = self._auth_client.set_session(access_token, refresh_token)
            
            if response.user and response.session:
                self._user = self._extract_user_dict(response.user)
                self._session = self._extract_session_dict(response.session)
                self._access_token = self._session.get('access_token')
                logger.info("Session restored")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Session restoration failed'}
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Session restore error: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_session_tokens(self) -> Optional[Dict[str, str]]:
        """Get current session tokens for storage."""
        if self._session:
            return {
                'access_token': self._session.get('access_token', ''),
                'refresh_token': self._session.get('refresh_token', '')
            }
        return None
    
    def check_subscription(self) -> Dict[str, Any]:
        """
        Check if current user has an active subscription.
        
        Returns:
            dict with 'has_subscription' bool, 'status', and 'expires_at'
        """
        if not self._user:
            logger.warning("[SUB CHECK] Not authenticated - no user")
            return {'has_subscription': False, 'status': None, 'error': 'Not authenticated'}
        
        try:
            user_id = self._user.get('id')
            logger.info(f"[SUB CHECK] Checking subscription for user_id: {user_id}")
            
            if not user_id:
                logger.warning("[SUB CHECK] No user ID found")
                return {'has_subscription': False, 'status': None, 'error': 'No user ID'}
            
            # Get DB client with auth token
            db_client = self._get_db_client()
            if not db_client:
                logger.warning("[SUB CHECK] Database client not available")
                return {'has_subscription': False, 'status': None, 'error': 'Database not available'}
            
            # Query subscriptions table
            logger.info(f"[SUB CHECK] Querying subscriptions table for user_id: {user_id}")
            response = db_client.from_('subscriptions').select('*').eq('user_id', user_id).execute()
            
            logger.info(f"[SUB CHECK] Query response: {response.data}")
            
            if response.data and len(response.data) > 0:
                sub = response.data[0]
                self._subscription = sub
                logger.info(f"[SUB CHECK] Found subscription: {sub}")
                
                status = sub.get('status')
                is_active = status in ('active', 'trialing')
                logger.info(f"[SUB CHECK] Status: {status}, is_active (before date check): {is_active}")
                
                # Check if subscription has expired
                period_end = sub.get('current_period_end')
                logger.info(f"[SUB CHECK] current_period_end: {period_end}")
                
                if period_end and is_active:
                    try:
                        end_date = datetime.fromisoformat(period_end.replace('Z', '+00:00'))
                        now = datetime.now(end_date.tzinfo)
                        logger.info(f"[SUB CHECK] end_date: {end_date}, now: {now}")
                        if end_date < now:
                            logger.info("[SUB CHECK] Subscription has EXPIRED")
                            is_active = False
                        else:
                            logger.info("[SUB CHECK] Subscription is VALID")
                    except Exception as e:
                        logger.warning(f"[SUB CHECK] Date parsing error: {e}")
                
                logger.info(f"[SUB CHECK] Final result: has_subscription={is_active}")
                return {
                    'has_subscription': is_active,
                    'status': status,
                    'expires_at': period_end
                }
            else:
                logger.warning(f"[SUB CHECK] No subscription found for user_id: {user_id}")
                return {'has_subscription': False, 'status': None}
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[SUB CHECK] Exception: {error_msg}")
            return {'has_subscription': False, 'status': None, 'error': error_msg}
    
    def open_checkout(self, price_id: str = None) -> bool:
        """
        Open Stripe checkout in browser for subscription.
        
        Args:
            price_id: Optional specific price ID (defaults to starter plan)
        
        Returns:
            True if browser was opened successfully
        """
        if not self._user:
            logger.warning("Cannot open checkout: user not authenticated")
            return False
        
        email = self._user.get('email', '')
        user_id = self._user.get('id', '')
        checkout_price = price_id or STRIPE_PRICE_ID
        
        # Create checkout URL with user info
        # This will redirect to our Supabase Edge Function that creates a Stripe Checkout Session
        checkout_url = f"{SUPABASE_URL}/functions/v1/create-checkout?user_id={user_id}&email={email}&price_id={checkout_price}&success_url={REDIRECT_URL_PAYMENT_SUCCESS}"
        
        try:
            webbrowser.open(checkout_url)
            logger.info(f"Opened checkout for user: {email}, price: {checkout_price}")
            return True
        except Exception as e:
            logger.error(f"Failed to open checkout: {e}")
            return False
    
    def get_plan_tier(self) -> str:
        """
        Get current user's plan tier based on their subscription price_id.
        
        Returns:
            'ultra' for Ultra plan, 'starter' for Starter plan, 'free' for no subscription
        """
        if not self._subscription:
            # Try to refresh subscription info
            self.check_subscription()
        
        if not self._subscription:
            return 'free'
        
        price_id = self._subscription.get('price_id', '')
        status = self._subscription.get('status', '')
        
        if status not in ('active', 'trialing'):
            return 'free'
        
        if price_id == STRIPE_PRICE_ID_ULTRA:
            return 'ultra'
        else:
            return 'starter'
    
    def get_index_limit(self) -> int:
        """Get the index limit for current user's plan (images, videos, audio only)."""
        tier = self.get_plan_tier()
        if tier == 'ultra':
            return INDEX_LIMIT_ULTRA
        elif tier == 'starter':
            return INDEX_LIMIT_STARTER
        else:
            return 0  # No subscription = no indexing
    
    def get_current_period_start(self) -> Optional[str]:
        """Get current billing period start date."""
        if not self._subscription:
            self.check_subscription()
        
        if self._subscription:
            return self._subscription.get('current_period_start')
        return None
    
    def get_index_usage(self) -> Dict[str, Any]:
        """
        Get current index usage for the billing period.
        
        Returns:
            dict with 'count', 'limit', 'remaining', 'period_start'
        """
        if not self._user:
            return {'count': 0, 'limit': 0, 'remaining': 0, 'error': 'Not authenticated'}
        
        try:
            user_id = self._user.get('id')
            period_start = self.get_current_period_start()
            limit = self.get_index_limit()
            
            if not period_start:
                # No subscription, no usage tracking
                return {'count': 0, 'limit': limit, 'remaining': limit}
            
            db_client = self._get_db_client()
            if not db_client:
                return {'count': 0, 'limit': limit, 'remaining': limit, 'error': 'Database not available'}
            
            # Query index_usage table for current period - just filter by user_id
            # (period_start comparison can fail due to timestamp format differences)
            response = db_client.from_('index_usage').select('*').eq(
                'user_id', user_id
            ).execute()
            
            logger.debug(f"[USAGE] Query response for user {user_id}: {response.data}")
            
            if response.data and len(response.data) > 0:
                # Find the record that matches our period (compare date portion only)
                target_date = period_start[:10] if period_start else None  # Extract YYYY-MM-DD
                count = 0
                for record in response.data:
                    record_period = record.get('period_start', '')
                    # Compare date portion to handle format differences
                    if record_period and target_date and record_period[:10] == target_date:
                        count = record.get('indexed_count', 0)
                        logger.debug(f"[USAGE] Found matching record: count={count}")
                        break
                    # Fallback: use most recent record if no exact match
                    if not count:
                        count = record.get('indexed_count', 0)
            else:
                count = 0
            
            logger.debug(f"[USAGE] Returning count={count}, limit={limit}")
            return {
                'count': count,
                'limit': limit,
                'remaining': max(0, limit - count),
                'period_start': period_start
            }
            
        except Exception as e:
            logger.error(f"Error getting index usage: {e}")
            return {'count': 0, 'limit': 0, 'remaining': 0, 'error': str(e)}
    
    def increment_index_usage(self, count: int = 1) -> bool:
        """
        Increment the index usage count for current billing period.
        
        Args:
            count: Number to increment by (default 1)
            
        Returns:
            True if successful
        """
        if not self._user:
            logger.warning("Cannot increment usage: not authenticated")
            return False
        
        try:
            user_id = self._user.get('id')
            period_start = self.get_current_period_start()
            
            if not period_start:
                logger.warning("Cannot increment usage: no subscription period")
                return False
            
            db_client = self._get_db_client()
            if not db_client:
                logger.warning("Cannot increment usage: database not available")
                return False
            
            # Query all records for this user
            response = db_client.from_('index_usage').select('*').eq(
                'user_id', user_id
            ).execute()
            
            now = datetime.now().isoformat()
            target_date = period_start[:10]  # Extract YYYY-MM-DD for comparison
            
            # Find matching record by date portion
            existing_record = None
            for record in (response.data or []):
                record_period = record.get('period_start', '')
                if record_period and record_period[:10] == target_date:
                    existing_record = record
                    break
            
            if existing_record:
                # Update existing record
                current_count = existing_record.get('indexed_count', 0)
                record_id = existing_record.get('id')
                db_client.from_('index_usage').update({
                    'indexed_count': current_count + count,
                    'updated_at': now
                }).eq('id', record_id).execute()
                logger.info(f"Updated index usage: {current_count} + {count} = {current_count + count}")
            else:
                # Insert new record
                db_client.from_('index_usage').insert({
                    'user_id': user_id,
                    'period_start': period_start,
                    'indexed_count': count,
                    'created_at': now,
                    'updated_at': now
                }).execute()
                logger.info(f"Created new index usage record with count={count}")
            
            logger.info(f"Incremented index usage by {count} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error incrementing index usage: {e}")
            return False
    
    def can_index_media(self, count: int = 1) -> Dict[str, Any]:
        """
        Check if user can index more images/videos.
        
        Args:
            count: Number of files to index
            
        Returns:
            dict with 'allowed', 'remaining', 'limit', 'plan'
        """
        tier = self.get_plan_tier()
        
        if tier == 'free':
            return {
                'allowed': False,
                'remaining': 0,
                'limit': 0,
                'plan': 'free',
                'reason': 'Subscription required to index files'
            }
        
        usage = self.get_index_usage()
        remaining = usage.get('remaining', 0)
        limit = usage.get('limit', 0)
        
        if remaining >= count:
            return {
                'allowed': True,
                'remaining': remaining,
                'limit': limit,
                'plan': tier
            }
        else:
            return {
                'allowed': False,
                'remaining': remaining,
                'limit': limit,
                'plan': tier,
                'reason': f'Index limit reached ({limit} files/month). Upgrade to Pro for more.'
            }
    
    def open_upgrade_checkout(self) -> bool:
        """Open checkout for upgrading to Ultra plan."""
        return self.open_checkout(price_id=STRIPE_PRICE_ID_ULTRA)


def get_latest_app_version() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest app version from Supabase (public access, no auth required).
    
    Filters by platform to ensure Windows users get Windows installers and
    Mac users get Mac installers.
    
    Returns:
        Dict with version info: {version, download_url, release_notes, release_name, is_required}
        or None if failed
    """
    import sys
    
    if not SUPABASE_AVAILABLE:
        logger.warning("Supabase not available for version check")
        return None
    
    # Determine current platform
    if sys.platform == 'win32':
        current_platform = 'windows'
    elif sys.platform == 'darwin':
        current_platform = 'mac'
    else:
        current_platform = 'linux'
    
    try:
        # Create anonymous client (no auth needed due to RLS policy)
        client = SyncPostgrestClient(
            base_url=f"{SUPABASE_URL}/rest/v1",
            headers={"apikey": SUPABASE_ANON_KEY}
        )
        
        # Query latest version for this platform (order by published_at desc, limit 1)
        response = (
            client.from_("app_version")
            .select("*")
            .eq("platform", current_platform)
            .order("published_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if response.data and len(response.data) > 0:
            version_data = response.data[0]
            logger.info(f"Fetched app version from Supabase: {version_data.get('version')} (platform: {current_platform})")
            return {
                'version': version_data.get('version'),
                'download_url': version_data.get('download_url'),
                'release_notes': version_data.get('release_notes', ''),
                'release_name': version_data.get('release_name', ''),
                'published_at': version_data.get('published_at', ''),
                'is_required': version_data.get('is_required', False)
            }
        else:
            logger.info(f"No app version found in Supabase for platform: {current_platform}")
            return None
            
    except Exception as e:
        logger.info(f"Could not fetch app version from Supabase: {e}")
        return None


# Global instance
supabase_auth = SupabaseAuth()
