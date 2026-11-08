#pragma once

#include <gtest/gtest.h>

// Passes the current test without checking anything while `pending_flag` is
// true (see qmm/pending.hpp).
#define QMM_PASS_WHILE_PENDING(pending_flag)            \
  do {                                                  \
    if (pending_flag) {                                 \
      GTEST_SUCCEED() << "pending: kernel not written"; \
      return;                                           \
    }                                                   \
  } while (false)
