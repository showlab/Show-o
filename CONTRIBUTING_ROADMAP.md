# Contributing to the Show-o Project

The Show-o project is open-sourced to the community to push the boundary of unified multimodal models. We invite you to join this exciting journey and contribute to the Show-o project!

## Submitting a Pull Request (PR)

As a contributor, before submitting your request, kindly follow these guidelines:

1. Start by checking the [Show-o GitHub](https://github.com/showlab/Show-o/pulls) to see if there are any open or closed pull requests related to your intended submission. Avoid duplicating existing work.

2. [Fork](https://github.com/showlab/Show-o/fork) the [Show-o](https://github.com/showlab/Show-o) repository and download your forked repository to your local machine.

   ```bash
   git clone [your-forked-repository-url]
   ```

3. Add the original repository as a remote to sync with the latest updates:

   ```bash
   git remote add upstream https://github.com/showlab/Show-o
   ```

4. Sync the code from the main repository to your local machine, and then push it back to your forked remote repository.

   ```
   # Pull the latest code from the upstream branch
   git fetch upstream
   
   # Switch to the main branch
   git checkout main
   
   # Merge the updates from the upstream branch into main, synchronizing the local main branch with the upstream
   git merge upstream/main
   
   # Additionally, sync the local main branch to the remote branch of your forked repository
   git push origin main
   ```


   > Note: Sync the code from the main repository before each submission.

5. Create a branch in your forked repository for your changes, ensuring the branch name is meaningful.

   ```bash
   git checkout -b my-docs-branch main
   ```

6. While making modifications and committing changes, adhere to our [Commit Message Format](#Commit-Message-Format).

   ```bash
   git commit -m "[docs]: xxxx"
   ```

7. Push your changes to your GitHub repository.

   ```bash
   git push origin my-docs-branch
   ```

8. Submit a pull request to `Show-o:main` on the GitHub repository page.

## Commit Message Format

Commit messages must include both `<type>` and `<summary>` sections.

```bash
[<type>]: <summary>
  │        │
  │        └─⫸ Briefly describe your changes, without ending with a period.
  │
  └─⫸ Commit Type: |docs|feat|fix|refactor|
```

### Type 

* **docs**: Modify or add documents.
* **feat**: Introduce a new feature.
* **fix**: Fix a bug.
* **refactor**: Restructure code, excluding new features or bug fixes.

### Summary

Describe modifications in English, without ending with a period.

> e.g., git commit -m "[docs]: add a contributing.md file"

## Roadmap
- 🛠️ Mixed-modal generation. (In progress by [@hrodruck](https://github.com/hrodruck))
- 🛠️ Support more modalities. (In progress by by [@LJungang](https://github.com/LJungang))
- 🛠️ Efficient training/inference. (In progress by [@KevinZeng08](https://github.com/KevinZeng08))
- 📣 Support training on more datasets. (Help wanted!)
- 📣 Visual tokenizer training. (Help wanted!)

### Acknowledgement
This guideline is modified from [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main) and [minisora](https://github.com/mini-sora/minisora). Thanks for their awesome templates.
