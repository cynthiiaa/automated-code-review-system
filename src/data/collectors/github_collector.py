import asyncio
from typing import List, Dict
from github import Github
import pandas as pd
from dataclasses import dataclass

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

    async def collect_pr_data(self, repo_name: str, limit: int = 100):
        """Collect PR data with existing reviews for training"""
        repo = self.client.get_repo(repo_name)
        pulls = repo.get_pulls(state="closed", sort="created", direction="desc")

        reviews = []
        for pr in pulls[:limit]:
            if pr.merged:
                review_data = await self._extract_review_data(pr)
                reviews.append(review_data)

        return reviews