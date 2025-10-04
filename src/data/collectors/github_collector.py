from typing import List, Dict, Optional
from github import Github
from github.PullRequest import PullRequest
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CodeReview:
    repo: str
    pr_number: int
    diff: str
    files_changed: List[str]
    language: str
    existing_comments: List[str]

class GitHubCollector:
    def __init__(self, token: str):
        self.client = Github(token)

    def collect_pr_data(self, repo_name: str, limit: int = 100) -> List[CodeReview]:
        """Collect PR data with existing reviews for training"""
        try:
            repo = self.client.get_repo(repo_name)
            pulls = repo.get_pulls(state="closed", sort="created", direction="desc")

            reviews = []
            for pr in pulls[:limit]:
                if pr.merged:
                    review_data = self._extract_review_data(pr)
                    reviews.append(review_data)

            return reviews
        except Exception as e:
            logger.error(f"Error collecting PR data from {repo_name}: {e}")
            return []
        
    def _extract_review_data(self, pr: PullRequest) -> Optional[CodeReview]:
        """Extract review data from a pull request"""
        try:
            # get diff and files
            files = pr.get_files()
            files_changed = [f.filename for f in files]

            # get the diff content
            diff_parts = []
            for file in files:
                if file.patch:
                    diff_parts.append(f"--- {file.filename} ---\n{file.patch}")
            diff = "\n\n".join(diff_parts)

            # get review comments
            comments = []
            for comment in pr.get_review_comments():
                comments.append(comment.body)

            # determine primary language from file extensions
            language = self._determine_language(files_changed)

            return CodeReview(
                repo=pr.base.repo.full_name,
                pr_number=pr.number,
                diff=diff,
                files_changed=files_changed,
                language=language,
                existing_comments=comments
            )
        except Exception as e:
            logger.error(f"Error extracting review data from PR #{pr.number}: {e}")
            return None
        
    def _determine_language(self, files: List[str]) -> str:
        """Determine the primary programming language from file extensions"""
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.cs': 'C#',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }

        language_counts = {}
        for file in files:
            for ext, lang in language_map.items():
                if file.endswith(ext):
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    break
        
        if language_counts:
            return max(language_counts, key=language_counts.get)
        return 'Unknown'
