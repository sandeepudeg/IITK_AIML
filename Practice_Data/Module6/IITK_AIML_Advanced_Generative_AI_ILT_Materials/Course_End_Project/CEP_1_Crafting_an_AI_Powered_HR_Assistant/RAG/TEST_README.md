# HR Policy Chatbot - Test Script

This test script validates the core functionality of the HR Policy Chatbot system.

## What It Tests

The test script covers the following requirements:

- **Requirement 1.1**: Document loading with PyPDFLoader
- **Requirement 1.2**: Document splitting into chunks
- **Requirement 1.3**: Error handling for invalid PDFs
- **Requirement 3.1**: Question answering with QA chain
- **Requirement 3.4**: Answer generation based on context
- **Requirement 3.5**: Handling unanswerable questions

## Test Suites

### 1. Document Loading Tests
- Load valid PDF documents
- Handle non-existent files
- Handle invalid file types
- Load multiple documents simultaneously

### 2. Document Splitting Tests
- Split documents with default parameters
- Handle empty document lists
- Validate chunk parameters

### 3. Error Handling Tests
- Vector store with empty documents
- Vector store with invalid API key
- Retriever with invalid k values

### 4. Question Answering Tests (requires API key)
- Create vector store and retriever
- Create QA chain
- Answer questions with available information
- Handle unanswerable questions
- Handle empty questions

## Running the Tests

### Basic Usage

```bash
python test_hr_chatbot.py
```

### With API Key

To run all tests including QA chain tests, ensure you have a valid OpenAI API key:

1. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. Run the test script:
   ```bash
   python test_hr_chatbot.py
   ```

### Without API Key

The script will automatically skip tests that require API access and run only the tests that don't need it:
- Document loading tests
- Document splitting tests
- Error handling tests

## Optional Dependencies

For full PDF testing functionality, install `reportlab`:

```bash
pip install reportlab
```

Without `reportlab`, the script will skip PDF creation tests but will still test error handling for invalid PDFs.

## Expected Output

The test script provides clear output:
- ✅ PASS: Test passed successfully
- ❌ FAIL: Test failed with error details
- ⚠️ SKIP: Test skipped (missing dependencies or API key)

At the end, a summary shows:
- Total tests run
- Number passed
- Number failed
- Details of any failures

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed

## Notes

- Tests create temporary files (`test_sample.pdf`, `test_doc1.pdf`, etc.) which are automatically cleaned up
- QA chain tests use sample HR policy content to validate functionality
- The script validates both success cases and error handling
- All tests are designed to be minimal and focused on core functionality
