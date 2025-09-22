package ro.ubb.ai.javaserver.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import ro.ubb.ai.javaserver.dto.user.UserDTO;
import ro.ubb.ai.javaserver.dto.user.UserUpdateRequest;
import ro.ubb.ai.javaserver.service.UserService;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/me")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<UserDTO> getCurrentUser() {
        return ResponseEntity.ok(userService.getCurrentUser());
    }

    @PutMapping("/{id}/settings")
    @PreAuthorize("isAuthenticated() and #id == authentication.principal.id")
    public ResponseEntity<UserDTO> updateUserSettings(@PathVariable Long id, @Valid @RequestBody UserUpdateRequest updateRequest) {
        UserDTO updatedUser = userService.updateUser(id, updateRequest);
        return ResponseEntity.ok(updatedUser);
    }
}
